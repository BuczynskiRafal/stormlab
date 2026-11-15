"""Uploads in another encoding reach every reader as the same UTF-8 bytes."""

from __future__ import annotations

from pathlib import Path

from sa.core.inp_encoding import source_encoding, utf8_chunks

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
POLISH = "[TITLE]\nOsiedle Łąkowe – zażółć gęślą jaźń\n[JUNCTIONS]\nWęzeł_ż 100 2\n"


def chunked(data: bytes, size: int):
    return lambda: (data[i : i + size] for i in range(0, len(data), size))


def converted(data: bytes, size: int = 7) -> bytes:
    return b"".join(utf8_chunks(chunked(data, size)))


def test_utf8_files_pass_through_byte_for_byte_even_when_a_chunk_splits_a_character():
    data = POLISH.encode("utf-8")
    for size in (1, 2, 3, 5, 4096):
        assert source_encoding(chunked(data, size)) == "utf-8"
        assert converted(data, size) == data


def test_every_shipped_example_is_passed_through_unchanged():
    for path in EXAMPLES.glob("*.inp"):
        data = path.read_bytes()
        assert converted(data, 4096) == data, path.name


def test_windows_1250_is_transcoded_with_the_polish_letters_intact():
    data = POLISH.encode("cp1250")

    assert source_encoding(chunked(data, 7)) == "cp1250"
    assert converted(data).decode("utf-8") == POLISH


def test_bytes_cp1250_leaves_undefined_fall_back_to_latin_1():
    data = b"[TITLE]\nx\x81y\x98z\n"

    assert source_encoding(chunked(data, 3)) == "latin-1"
    assert converted(data).decode("utf-8") == "[TITLE]\nx\x81y\x98z\n"


def test_line_breaks_and_nul_bytes_survive_transcoding():
    data = "a\r\nż\x00b\n".encode("cp1250")

    assert converted(data, 1) == "a\r\nż\x00b\n".encode("utf-8")
