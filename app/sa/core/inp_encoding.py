"""Give every reader of an uploaded model the same UTF-8 text, whatever encoding the file came in.

swmmio and the feature extraction read the model as UTF-8 and fail on any other byte
sequence, while the SWMM engine reads plain bytes. Files written by the Windows SWMM
GUI use the system code page, which for the Polish designers this application is for is
Windows-1250: "ł" is the single byte ``0xB3`` there and an invalid UTF-8 sequence.

So an upload that is not valid UTF-8 is transcoded once, when it is copied, and every
later step (the safety scan, the upload gates, the engine, swmmio) reads the same UTF-8
bytes:

* **cp1250** is tried first: it is what SWMM on a Polish Windows writes, and it decodes
  "ąćęłńóśźż" correctly. ISO-8859-2 differs from it in exactly some of those letters, so
  the guess cannot be made safely the other way round.
* **latin-1** is the fallback for the five bytes cp1250 leaves undefined. It maps every
  byte to one character, so it never fails and keeps the file's structure (section
  headers, separators and digits are ASCII in every one of these encodings) and every
  name spelled the same way wherever it occurs.

A file that already is valid UTF-8 (ASCII included) is passed through byte for byte.
"""

from __future__ import annotations

import codecs
from collections.abc import Callable, Iterable, Iterator

FALLBACK_ENCODINGS = ("cp1250", "latin-1")


def _decodes(chunks: Iterable[bytes], encoding: str) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    try:
        for chunk in chunks:
            decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def source_encoding(open_chunks: Callable[[], Iterable[bytes]]) -> str:
    """``"utf-8"`` when the file decodes as UTF-8, else the first fallback that decodes it."""
    for encoding in ("utf-8", *FALLBACK_ENCODINGS[:-1]):
        if _decodes(open_chunks(), encoding):
            return encoding
    return FALLBACK_ENCODINGS[-1]


def _transcode(chunks: Iterable[bytes], encoding: str) -> Iterator[bytes]:
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    for chunk in chunks:
        yield decoder.decode(chunk).encode("utf-8")
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail.encode("utf-8")


def utf8_chunks(open_chunks: Callable[[], Iterable[bytes]]) -> Iterable[bytes]:
    """The file as UTF-8 byte chunks; *open_chunks* is called once or twice, each time from the start."""
    encoding = source_encoding(open_chunks)
    if encoding == "utf-8":
        return open_chunks()
    return _transcode(open_chunks(), encoding)
