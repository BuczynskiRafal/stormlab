"""The numbers.json schema: no NaN, explicit infinities, unique keys."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stormlab.contracts.numbers import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_FIELD,
    RecordFactory,
    DuplicateNumberKeyError,
    NumberRecord,
    NumbersRegistry,
    NumberValidationError,
)


def _record(key: str = "tabular.mlp.test.macro_f1", value: float = 0.873, **overrides: object) -> NumberRecord:
    payload: dict[str, object] = {
        "key": key,
        "value": value,
        "unit": "1",
        "kind": "metric",
        "producer": "experiments.e_tabular",
        "run_id": "20260827T120000Z",
        "git_commit": "7757efc",
        "tolerance": 0.001,
        "population": "test_1254_rows/114_files",
        "model_id": "mlp_retrained_v1",
        "class_order": "recommendation_enum",
        "thesis_ref": "Tab. 3.16",
        "prob_rounding": "raw",
    }
    payload.update(overrides)
    return NumberRecord(**payload)  # type: ignore[arg-type]


def test_a_finite_record_round_trips_through_json() -> None:
    record = _record()
    restored = NumberRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored == record
    assert restored.is_inf is False


def test_nan_is_rejected_at_construction() -> None:
    with pytest.raises(NumberValidationError, match="is NaN"):
        _record(value=math.nan)


def test_infinity_is_stored_as_a_string_with_an_explicit_flag() -> None:
    record = _record(key="tabular.svm.test.ce_all9", value=math.inf)
    payload = record.to_dict()
    assert payload["value"] == "inf"
    assert payload["is_inf"] is True
    assert NumberRecord.from_dict(payload).value == math.inf


def test_negative_infinity_keeps_its_sign() -> None:
    payload = _record(value=-math.inf).to_dict()
    assert payload["value"] == "-inf"
    assert NumberRecord.from_dict(payload).value == -math.inf


def test_a_flag_disagreeing_with_the_value_is_rejected() -> None:
    payload = _record().to_dict()
    payload["is_inf"] = True
    with pytest.raises(NumberValidationError, match="disagrees"):
        NumberRecord.from_dict(payload)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="Unknown kind"):
        _record(kind="guess")


def test_unknown_prob_rounding_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="Unknown prob_rounding"):
        _record(prob_rounding="4dp")


def test_a_key_without_a_namespace_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="dot-separated"):
        _record(key="accuracy")


def test_registry_refuses_a_duplicate_key() -> None:
    registry = NumbersRegistry()
    registry.add(_record())
    with pytest.raises(DuplicateNumberKeyError, match="already registered"):
        registry.add(_record(value=0.9))


def test_registry_round_trips_through_a_file(tmp_path: Path) -> None:
    registry = NumbersRegistry()
    registry.add(_record())
    registry.add(_record(key="tabular.svm.test.ce_all9", value=math.inf))
    target = tmp_path / "numbers.json"
    registry.to_json(target)

    restored = NumbersRegistry.from_json(target)
    assert len(restored) == 2
    assert restored.get("tabular.svm.test.ce_all9").value == math.inf


def test_serialised_registry_is_sorted_by_key(tmp_path: Path) -> None:
    registry = NumbersRegistry()
    registry.add(_record(key="zzz.model.test.metric"))
    registry.add(_record(key="aaa.model.test.metric"))
    payload = json.loads(registry.to_json(tmp_path / "numbers.json"))
    assert [item["key"] for item in payload] == ["aaa.model.test.metric", "zzz.model.test.metric"]


def test_a_boolean_tolerance_is_rejected() -> None:
    """``True`` would pass an int check and quietly mean a tolerance of 1.0."""
    with pytest.raises(NumberValidationError, match="must be a real number"):
        _record(tolerance=True)


def test_a_negative_tolerance_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="non-negative"):
        _record(tolerance=-1e-9)


def test_an_infinite_tolerance_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="finite"):
        _record(tolerance=float("inf"))


def test_a_nan_tolerance_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="finite"):
        _record(tolerance=float("nan"))


def test_a_non_numeric_tolerance_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="must be a real number"):
        _record(tolerance="1e-9")  # type: ignore[arg-type]


def test_a_zero_tolerance_is_accepted() -> None:
    assert _record(tolerance=0.0).tolerance == 0.0


def test_a_note_carries_text_instead_of_a_figure() -> None:
    """Some statements of the thesis are not numbers; writing them as 0 would be a lie."""
    record = _record(key="dataset.historical_7128", value="unreproducible", kind="note")

    assert record.value == "unreproducible"
    assert record.is_inf is False
    assert NumberRecord.from_dict(record.to_dict()) == record


def test_an_empty_note_is_rejected() -> None:
    with pytest.raises(NumberValidationError, match="non-empty string"):
        _record(key="dataset.note", value="   ", kind="note")


def test_text_under_a_numeric_kind_is_still_rejected() -> None:
    with pytest.raises(NumberValidationError, match="must be a real number"):
        _record(value="quite high", kind="metric")


def test_a_named_parameter_may_carry_its_string_value() -> None:
    record = _record(key="split.group_v2.unit", value="super_lineage_id", kind="parameter")
    assert NumberRecord.from_dict(record.to_dict()) == record


def test_parameter_with_inf_round_trips() -> None:
    record = _record(key="split.some_threshold", value=math.inf, kind="parameter")
    restored = NumberRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.value == math.inf


def test_a_registry_holding_a_note_round_trips(tmp_path: Path) -> None:
    registry = NumbersRegistry()
    registry.add(_record())
    registry.add(_record(key="dataset.historical_7128", value="unreproducible", kind="note"))
    target = tmp_path / "numbers.json"
    registry.to_json(target)

    restored = NumbersRegistry.from_json(target)

    assert restored.get("dataset.historical_7128").value == "unreproducible"


class TestRecordFactory:
    """The factory exists so a study cannot register a figure that names no input."""

    def _factory(self) -> RecordFactory:
        return RecordFactory(run_id="r1", git_commit="deadbeef", input_sha256="a" * 64, producer="tests")

    def test_every_record_carries_the_runs_provenance(self) -> None:
        record = self._factory().make("study.metric", 0.5, kind="metric")

        assert (record.run_id, record.git_commit, record.producer) == ("r1", "deadbeef", "tests")
        assert record.input_sha256 == "a" * 64

    @pytest.mark.parametrize("field", ["run_id", "git_commit", "input_sha256", "producer"])
    def test_a_missing_provenance_field_is_refused_at_construction(self, field: str) -> None:
        values = {"run_id": "r1", "git_commit": "c", "input_sha256": "s", "producer": "p", field: ""}

        with pytest.raises(NumberValidationError, match=field):
            RecordFactory(**values)

    @pytest.mark.parametrize("field", ["run_id", "git_commit", "input_sha256", "producer"])
    def test_a_caller_cannot_override_what_the_factory_guarantees(self, field: str) -> None:
        with pytest.raises(NumberValidationError, match="fixed by the factory"):
            self._factory().make("study.metric", 1.0, kind="metric", **{field: "something else"})

    def test_a_prediction_record_must_state_its_convention(self) -> None:
        record = self._factory().prediction(
            "study.model.test.n_correct",
            42,
            kind="count",
            model_id="mlp_retrained_v1",
            class_order="recommendation_enum",
            prob_rounding="raw",
        )

        assert (record.model_id, record.class_order, record.prob_rounding) == (
            "mlp_retrained_v1",
            "recommendation_enum",
            "raw",
        )

    @pytest.mark.parametrize("omitted", ["model_id", "class_order", "prob_rounding"])
    def test_a_prediction_record_cannot_omit_the_convention(self, omitted: str) -> None:
        arguments = {"model_id": "m", "class_order": "o", "prob_rounding": "raw"}
        del arguments[omitted]

        with pytest.raises(TypeError, match=omitted):
            self._factory().prediction("study.model.test.value", 1.0, kind="count", **arguments)


class TestSchemaVersion:
    """Version 2 added ``variant``; version 1 registries still have to be readable."""

    def test_a_written_record_declares_the_current_schema_version(self) -> None:
        payload = _record(variant="B1").to_dict()

        assert payload[SCHEMA_VERSION_FIELD] == SCHEMA_VERSION
        assert payload["variant"] == "B1"

    def test_a_version_one_payload_reads_as_a_record_with_no_variant(self) -> None:
        payload = {key: value for key, value in _record().to_dict().items() if key not in {SCHEMA_VERSION_FIELD, "variant"}}

        restored = NumberRecord.from_dict(payload)

        assert restored.variant is None

    def test_a_version_two_payload_round_trips_with_its_variant(self) -> None:
        record = _record(variant="B2")

        assert NumberRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record

    def test_a_schema_version_this_package_cannot_read_is_refused(self) -> None:
        payload = _record().to_dict()
        payload[SCHEMA_VERSION_FIELD] = 99

        with pytest.raises(NumberValidationError, match="schema_version"):
            NumberRecord.from_dict(payload)

    @pytest.mark.parametrize("version", [True, 2.0], ids=["bool", "float"])
    def test_a_schema_version_that_is_not_an_integer_is_refused(self, version: object) -> None:
        """``True`` and ``2.0`` both compare equal to a supported version."""
        payload = _record().to_dict()
        payload[SCHEMA_VERSION_FIELD] = version

        with pytest.raises(NumberValidationError, match="must be an integer"):
            NumberRecord.from_dict(payload)

    def test_an_empty_variant_is_refused(self) -> None:
        with pytest.raises(NumberValidationError, match="Variant of"):
            _record(variant="  ")

    def test_the_factory_stamps_its_variant_on_every_record(self) -> None:
        factory = RecordFactory(run_id="r", git_commit="c", input_sha256="f" * 64, producer="tests", variant="B1")

        assert factory.make("a.b", 1.0, kind="count").variant == "B1"

    def test_the_factorys_variant_cannot_be_overridden_at_the_call_site(self) -> None:
        factory = RecordFactory(run_id="r", git_commit="c", input_sha256="f" * 64, producer="tests", variant="B1")

        with pytest.raises(NumberValidationError, match="fixed by the factory"):
            factory.make("a.b", 1.0, kind="count", variant="B2")
