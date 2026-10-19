"""Provenance records pin the environment and are never overwritten."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stormlab.provenance.record import (
    PIPELINE_INPUT_PATHS,
    TRACKED_LIBRARIES,
    ProvenanceError,
    ProvenanceRecord,
    git_state,
    library_versions,
    sha256_file,
    worktree_status,
)

EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _record(tmp_path: Path, step: str = "features", run_id: str = "run-1") -> ProvenanceRecord:
    source = tmp_path / "input.inp"
    source.write_text("[TITLE]\n", encoding="utf-8")
    return ProvenanceRecord.capture(step=step, run_id=run_id, seeds={"numpy": 42}, inputs={"inp": source})


def test_sha256_of_an_empty_file_matches_the_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sha256_file(target) == EMPTY_FILE_SHA256


def test_hashing_a_missing_file_raises_with_the_path(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="not an existing file"):
        sha256_file(tmp_path / "absent.bin")


def test_library_versions_cover_every_tracked_distribution() -> None:
    versions = library_versions()
    assert set(versions) == set(TRACKED_LIBRARIES)
    assert versions["numpy"] is not None
    assert versions["pandas"] is not None


def test_capture_records_commit_seeds_and_input_hashes(tmp_path: Path) -> None:
    record = _record(tmp_path)
    assert record.git_commit
    assert isinstance(record.git_dirty, bool)
    assert record.git_dirty_scope == list(PIPELINE_INPUT_PATHS)
    assert record.seeds == {"numpy": 42}
    assert len(record.inputs["inp"]) == 64
    assert record.python_version


def test_write_creates_run_id_and_step_named_json(tmp_path: Path) -> None:
    record = _record(tmp_path)
    target = record.write(tmp_path / "provenance")
    assert target == tmp_path / "provenance" / "run-1" / "features.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["step"] == "features"
    assert payload["run_id"] == "run-1"


def test_writing_the_same_step_twice_is_refused(tmp_path: Path) -> None:
    record = _record(tmp_path)
    root = tmp_path / "provenance"
    record.write(root)
    with pytest.raises(ProvenanceError, match="already exists"):
        record.write(root)


def test_a_different_step_of_the_same_run_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "provenance"
    _record(tmp_path, step="features").write(root)
    second = _record(tmp_path, step="labels").write(root)
    assert second.name == "labels.json"


def test_written_record_can_be_read_back(tmp_path: Path) -> None:
    record = _record(tmp_path)
    target = record.write(tmp_path / "provenance")
    assert ProvenanceRecord.read(target) == record


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b"])
def test_path_like_step_names_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ProvenanceError, match="path-free identifier"):
        _record(tmp_path, step=bad)


class TestDirtyIsAboutInputs:
    """ "Dirty" is a statement about what the run read, not about what it wrote.

    Every case runs against a throwaway repository rather than this one: the
    question is what ``git status`` is asked, and only a tree whose contents the
    test controls can answer it.
    """

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        import subprocess

        root = tmp_path / "repo"
        (root / "stormlab").mkdir(parents=True)
        (root / "results" / "ablation" / "B2").mkdir(parents=True)
        (root / "models").mkdir()
        (root / "stormlab" / "config.yaml").write_text("variant: B1\n", encoding="utf-8")
        (root / "results" / "ablation" / "B2" / "S1.csv").write_text("k,score\n1,0.5\n", encoding="utf-8")
        (root / "models" / "weights.h5").write_text("w\n", encoding="utf-8")
        run = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(["git", "init", "-q", "."], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run([*run, "commit", "-qm", "init"], cwd=root, check=True)
        return root

    def test_a_run_writing_its_own_results_does_not_make_the_tree_dirty(self, tmp_path: Path) -> None:
        """The defect: `update-golden` refused to freeze a variant because it had been computed."""
        root = self._repo(tmp_path)
        (root / "results" / "ablation" / "B2" / "S1.csv").write_text("k,score\n1,0.9\n", encoding="utf-8")
        (root / "results" / "ablation" / "B2" / "verdicts.json").write_text("{}\n", encoding="utf-8")

        commit, entries = worktree_status(root)

        assert entries == ()
        assert git_state(root) == (commit, False)

    def test_an_edit_to_the_code_makes_the_tree_dirty(self, tmp_path: Path) -> None:
        root = self._repo(tmp_path)
        (root / "stormlab" / "config.yaml").write_text("variant: B2\n", encoding="utf-8")

        _, entries = worktree_status(root)

        assert [entry for entry in entries if entry.endswith("stormlab/config.yaml")]
        assert git_state(root)[1] is True

    def test_an_untracked_model_makes_the_tree_dirty(self, tmp_path: Path) -> None:
        """A file git does not track is a file the next checkout will not have."""
        root = self._repo(tmp_path)
        (root / "models" / "new.keras").write_text("w\n", encoding="utf-8")

        _, entries = worktree_status(root)

        assert any(entry.startswith("??") for entry in entries)
        assert git_state(root)[1] is True

    def test_a_path_the_checkout_does_not_have_is_not_a_dirty_worktree(self, tmp_path: Path) -> None:
        """``app/`` is absent from some checkouts; absence is not an uncommitted change."""
        root = self._repo(tmp_path)

        assert "app" in PIPELINE_INPUT_PATHS
        assert not (root / "app").exists()
        assert worktree_status(root)[1] == ()

    def test_the_scope_is_written_into_the_record_it_qualifies(self, tmp_path: Path) -> None:
        record = _record(tmp_path)

        payload = json.loads(record.write(tmp_path / "provenance").read_text(encoding="utf-8"))

        assert payload["git_dirty_scope"] == list(PIPELINE_INPUT_PATHS)
        assert "results/runs" not in payload["git_dirty_scope"]

    def test_a_caller_may_narrow_the_scope_it_asks_about(self, tmp_path: Path) -> None:
        root = self._repo(tmp_path)
        (root / "stormlab" / "config.yaml").write_text("variant: B2\n", encoding="utf-8")

        assert git_state(root, paths=("models",))[1] is False
        assert git_state(root, paths=("stormlab",))[1] is True
