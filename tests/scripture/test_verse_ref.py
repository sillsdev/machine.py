import pytest

from machine.scripture import (
    ENGLISH_VERSIFICATION,
    LAST_BOOK,
    SEPTUAGINT_VERSIFICATION,
    VULGATE_VERSIFICATION,
    ValidStatus,
    VerseRef,
    Versification,
)
from machine.scripture.versification import (
    _LineType,
    _parse_excluded_verse,
    _VersificationLine,
)


def test_constructor() -> None:
    vref = VerseRef(1, 2, 3, SEPTUAGINT_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 1002003
    assert vref.bbbcccvvvs == "001002003"
    assert vref.book_num == 1
    assert vref.book == "GEN"
    assert vref.chapter_num == 2
    assert vref.chapter == "2"
    assert vref.verse_num == 3
    assert vref.versification == SEPTUAGINT_VERSIFICATION

    vref = VerseRef(4, 5, 6)
    assert vref.bbbcccvvv == 4005006
    assert vref.bbbcccvvvs == "004005006"
    assert vref.book_num == 4
    assert vref.book == "NUM"
    assert vref.chapter_num == 5
    assert vref.verse_num == 6
    assert vref.versification == ENGLISH_VERSIFICATION

    vref = VerseRef("LUK", "3", "4b-5a", VULGATE_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 42003004
    assert vref.bbbcccvvvs == "042003004b"
    assert vref.book_num == 42
    assert vref.chapter_num == 3
    assert vref.verse_num == 4
    assert vref.verse == "4b-5a"
    assert vref.segment() == "b"
    assert sum(1 for _ in vref.all_verses()) == 2
    assert vref.versification == VULGATE_VERSIFICATION

    # Confirm RTL marker is removed
    vref = VerseRef("LUK", "3", "4b\u200f-5a", VULGATE_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 42003004
    assert vref.bbbcccvvvs == "042003004b"
    assert vref.book_num == 42
    assert vref.chapter_num == 3
    assert vref.verse_num == 4
    assert vref.verse == "4b-5a"
    assert vref.segment() == "b"
    assert sum(1 for _ in vref.all_verses()) == 2
    assert vref.versification == VULGATE_VERSIFICATION


def test_from_string() -> None:
    vref = VerseRef.from_string("LUK 3:4b-5a", VULGATE_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 42003004
    assert vref.bbbcccvvvs == "042003004b"
    assert vref.book_num == 42
    assert vref.chapter_num == 3
    assert vref.verse_num == 4
    assert vref.verse == "4b-5a"
    assert vref.segment() == "b"
    assert sum(1 for _ in vref.all_verses()) == 2
    assert vref.versification == VULGATE_VERSIFICATION

    # Confirm RTL marker is removed
    vref = VerseRef.from_string("LUK 3\u200f:4\u200f-5", VULGATE_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 42003004
    assert vref.bbbcccvvvs == "042003004"
    assert vref.book_num == 42
    assert vref.chapter_num == 3
    assert vref.verse_num == 4
    assert vref.verse == "4-5"
    assert vref.segment() == ""
    assert sum(1 for _ in vref.all_verses()) == 2
    assert vref.versification == VULGATE_VERSIFICATION


def test_from_bbbcccvvv() -> None:
    vref = VerseRef.from_bbbcccvvv(12015013)
    assert vref.bbbcccvvv == 12015013
    assert vref.bbbcccvvvs == "012015013"
    assert vref.book == "2KI"
    assert vref.book_num == 12
    assert vref.chapter_num == 15
    assert vref.verse_num == 13
    assert vref.verse == "13"
    assert vref.versification == ENGLISH_VERSIFICATION


def test_chapter_and_verse_as_empty_strings() -> None:
    vref = VerseRef("LUK", "", "", SEPTUAGINT_VERSIFICATION)
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.book == "LUK"
    assert vref.chapter == ""
    assert vref.verse == ""
    assert vref.book_num == 42
    assert vref.chapter_num == -1
    assert vref.verse_num == -1

    vref = VerseRef("LUK", "5", "3", SEPTUAGINT_VERSIFICATION)
    vref.verse = ""
    vref.chapter = ""
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.chapter == ""
    assert vref.verse == ""
    assert vref.chapter_num == -1
    assert vref.verse_num == -1


def test_verse_with_rtl_markers() -> None:
    vref = VerseRef("LUK", "5", "1\u200f-2", SEPTUAGINT_VERSIFICATION)
    assert vref.valid_status == ValidStatus.VALID
    assert vref.book == "LUK"
    assert vref.chapter == "5"
    assert vref.verse == "1-2"
    assert vref.book_num == 42
    assert vref.chapter_num == 5
    assert vref.verse_num == 1


def test_build_verse_ref_by_props() -> None:
    vref = VerseRef()
    vref.versification = ENGLISH_VERSIFICATION
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.bbbcccvvv == 0

    vref.book_num = 13
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.bbbcccvvv == 13000000
    assert vref.book_num == 13
    assert vref.chapter_num == 0
    assert vref.verse_num == 0

    vref.chapter_num = 1
    vref.verse_num = 0
    # a zero verse is considered valid for introduction, etc, but only for chapter 1
    assert vref.is_valid
    assert vref.bbbcccvvv == 13001000
    assert vref.book_num == 13
    assert vref.chapter_num == 1
    assert vref.verse_num == 0

    vref.chapter_num = 14
    vref.verse_num = 15
    assert vref.is_valid
    assert vref.bbbcccvvv == 13014015
    assert vref.book_num == 13
    assert vref.chapter_num == 14
    assert vref.verse_num == 15

    vref = VerseRef()
    vref.versification = ENGLISH_VERSIFICATION
    vref.chapter_num = 16
    # Invalid because 0 is not valid for the book number
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.bbbcccvvv == 16000
    assert vref.book_num == 0
    assert vref.chapter_num == 16
    assert vref.verse_num == 0

    vref = VerseRef()
    vref.versification = ENGLISH_VERSIFICATION
    vref.verse_num = 17
    # Invalid because 0 is not valid for the book and chapter numbers
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.bbbcccvvv == 17
    assert vref.book_num == 0
    assert vref.chapter_num == 0
    assert vref.verse_num == 17


def test_invalid() -> None:
    with pytest.raises(ValueError):
        VerseRef(-1, 1, 1)
    with pytest.raises(ValueError):
        VerseRef(0, 1, 1)
    with pytest.raises(ValueError):
        VerseRef(LAST_BOOK + 1, 1, 1)
    with pytest.raises(ValueError):
        VerseRef(2, -42, 1)
    with pytest.raises(ValueError):
        VerseRef(2, 1, -4)
    with pytest.raises(ValueError):
        VerseRef.from_string("MAT 1:")
    with pytest.raises(ValueError):
        VerseRef.from_string("MAT 1:2-")
    with pytest.raises(ValueError):
        VerseRef.from_string("MAT 1:2,")

    vref = VerseRef(1, 1023, 5051, ENGLISH_VERSIFICATION)
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.book_num == 1
    assert vref.chapter_num == 1023
    assert vref.verse_num == 5051

    vref = VerseRef("GEN", "F", "@", ENGLISH_VERSIFICATION)
    assert vref.valid_status == ValidStatus.OUT_OF_RANGE
    assert vref.book_num == 1
    assert vref.chapter_num == -1
    assert vref.verse_num == -1


def test_segments() -> None:
    assert VerseRef.from_string("MAT 3:13").bbbcccvvvs == "040003013"
    assert VerseRef.from_string("MAT 3:12a").bbbcccvvvs == "040003012a"
    assert VerseRef.from_string("1KI 2:35a-35h").bbbcccvvvs == "011002035a"
    assert VerseRef.from_string("ESG 8:8a").bbbcccvvvs == "069008008a"
    assert VerseRef.from_string("MAT 12:1-3,5a,6c-9").bbbcccvvvs == "040012001"
    assert VerseRef.from_string("MAT 3:13b-12a").bbbcccvvvs == "040003013b"


def test_is_valid() -> None:
    assert VerseRef.from_string("GEN 1:1").is_valid
    assert VerseRef.from_string("GEN 1:1-2").is_valid
    assert VerseRef.from_string("GEN 1:1,3").is_valid
    assert VerseRef.from_string("GEN 1:1,3,7").is_valid
    assert VerseRef.from_string("PSA 119:1,3-6").is_valid


def test_is_valid_segments() -> None:
    assert VerseRef.from_string("GEN 1:1b").is_valid
    assert VerseRef.from_string("GEN 1:1c-2a").is_valid
    assert VerseRef.from_string("GEN 1:1a,3b").is_valid
    assert VerseRef.from_string("GEN 1:1a,3c,7b").is_valid
    assert VerseRef.from_string("GEN 1:1a,3c-6a").is_valid


def test_valid_status_invalid_order() -> None:
    assert VerseRef.from_string("GEN 1:2-1").valid_status == ValidStatus.VERSE_OUT_OF_ORDER
    assert VerseRef.from_string("GEN 1:2,1").valid_status == ValidStatus.VERSE_OUT_OF_ORDER
    assert VerseRef.from_string("GEN 1:2-3,1").valid_status == ValidStatus.VERSE_OUT_OF_ORDER
    assert VerseRef.from_string("GEN 1:5,2-3").valid_status == ValidStatus.VERSE_OUT_OF_ORDER


def test_valid_status_invalid_in_versification() -> None:
    # Invalid chapters
    assert VerseRef.from_string("GEN 100:1").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("PHM 2:1").valid_status == ValidStatus.OUT_OF_RANGE

    # Invalid verses
    assert VerseRef.from_string("GEN 1:100").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:100-2").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1-200").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:100,3").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1,300").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:100,3,7").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1,300,7").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1,3,700").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:100,3-6").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1,300-6").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1,3-600").valid_status == ValidStatus.OUT_OF_RANGE


def test_valid_status_invalid_excluded_in_versification() -> None:
    versification = Versification.create("Dummy")
    versification._excluded_verses.add(VerseRef.from_string("GEN 1:30").bbbcccvvv)

    # Valid verses (surrounding excluded verse)
    assert VerseRef.from_string("GEN 1:29", versification).is_valid
    assert VerseRef.from_string("GEN 1:31", versification).is_valid

    # Invalid (excluded) verse
    assert VerseRef.from_string("GEN 1:30", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:30,31", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:29,30", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:29-30", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:30-31", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:30b", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:30b-31a", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:29b-30a", versification).valid_status == ValidStatus.OUT_OF_RANGE
