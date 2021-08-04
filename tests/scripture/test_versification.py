from io import StringIO

import pytest

from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, Versification


def test_parse_valid() -> None:
    src = (
        '# Versification  "Test"\n'
        "# Version=1.9\n"
        "GEN 1:31 2:25 3:24 4:26 5:32 6:22 7:24 8:22 9:29 10:32 11:32 12:20 13:18 14:24 15:21 16:16 17:27 18:33 19:38 20:18 21:34 22:24 23:20 24:67\n"
        "MRK 1:45 2:28 3:35 4:41 5:44 6:56\n"
        "MRK 5:44 = MRK 6:1\n"
    )
    stream = StringIO(src)
    versification = Versification.parse(stream, "vers.txt")
    assert versification.get_last_book() == 41
    assert versification.get_last_chapter(1) == 24
    assert versification.get_last_chapter(41) == 6
    assert versification.get_last_verse(41, 2) == 28

    reference = VerseRef.from_bbbcccvvv(41005044, versification)
    reference.change_versification(ORIGINAL_VERSIFICATION)
    assert reference.bbbcccvvv == 41006001


def test_parse_without_name() -> None:
    src = "GEN 1:31\n" "MRK 1:45\n"
    stream = StringIO(src)
    with pytest.raises(RuntimeError):
        Versification.parse(stream, "vers.txt")


def test_parse_invalid_syntax() -> None:
    src = "GEN 1:31 MRK 1:-8MAT 5:44 = FFF6,1\n"
    stream = StringIO(src)
    with pytest.raises(RuntimeError):
        Versification.parse(stream, "vers.txt", fallback_name="name")
