import pytest

from machine.scripture import (
    ENGLISH_VERSIFICATION,
    LAST_BOOK,
    ORIGINAL_VERSIFICATION,
    RUSSIAN_ORTHODOX_VERSIFICATION,
    SEPTUAGINT_VERSIFICATION,
    VULGATE_VERSIFICATION,
    ValidStatus,
    VerseRef,
    Versification,
    get_bbbcccvvv,
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
    assert vref.validated_segment() == "b"
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
    assert vref.validated_segment() == "b"
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
    assert vref.validated_segment() == "b"
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
    assert vref.validated_segment() == ""
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
    versification.excluded_verses.add(VerseRef.from_string("GEN 1:30").bbbcccvvv)

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


def test_valid_status_excluded_verse() -> None:
    versification = Versification.create("Dummy")
    versification.excluded_verses.add(get_bbbcccvvv(1, 2, 2))

    # If an excluded verse is within a verse range, it is valid.
    assert VerseRef.from_string("GEN 2:1-3", versification).is_valid

    # If an excluded verse is explicitly included in the reference, it is invalid.
    assert VerseRef.from_string("GEN 2:2", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 2:2-3", versification).valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 2:1-2", versification).valid_status == ValidStatus.OUT_OF_RANGE


def test_valid_status_invalid_versification_on_segments() -> None:
    assert VerseRef.from_string("GEN 1:100b").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1c-200a").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1a,300b").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1a,3c,700b").valid_status == ValidStatus.OUT_OF_RANGE
    assert VerseRef.from_string("GEN 1:1a,3c-600a").valid_status == ValidStatus.OUT_OF_RANGE


def test_from_string_valid() -> None:
    vref = VerseRef.from_string("Gen 1:1", ENGLISH_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 1001001


def test_from_string_bridge() -> None:
    vref = VerseRef.from_string("NUM 5:1-5", ENGLISH_VERSIFICATION)
    assert vref.is_valid
    assert vref.bbbcccvvv == 4005001
    assert vref.bbbcccvvvs == "004005001"
    assert str(vref) == "NUM 5:1-5"
    assert vref.str_with_versification() == "NUM 5:1-5/4"
    assert vref.book_num == 4
    assert vref.chapter_num == 5
    assert vref.verse_num == 1
    assert vref.verse == "1-5"
    assert vref.versification == ENGLISH_VERSIFICATION


def test_from_string_bridge_with_versification() -> None:
    vref = VerseRef.from_string("NUM 5:1-5/2")
    assert vref.is_valid
    assert vref.bbbcccvvv == 4005001
    assert vref.bbbcccvvvs == "004005001"
    assert str(vref) == "NUM 5:1-5"
    assert vref.str_with_versification() == "NUM 5:1-5/2"
    assert vref.book_num == 4
    assert vref.chapter_num == 5
    assert vref.verse_num == 1
    assert vref.verse == "1-5"
    assert vref.versification == SEPTUAGINT_VERSIFICATION


def test_from_string_book_intro() -> None:
    vref = VerseRef.from_string("JOS 1:0")
    assert vref.is_valid
    assert vref.bbbcccvvv == 6001000


def test_from_string_chapter_intro() -> None:
    vref = VerseRef.from_string("JOS 2:0")
    assert vref.is_valid
    assert vref.bbbcccvvv == 6002000


def test_from_string_weird() -> None:
    vref = VerseRef.from_string("EXO 0:18")
    assert not vref.is_valid
    assert vref.bbbcccvvv == 2000018
    assert vref.book_num == 2
    assert vref.chapter_num == 0
    assert vref.verse_num == 18


def test_parse_ref_invalid_book() -> None:
    with pytest.raises(ValueError):
        VerseRef.from_string("BLA 1:1")
    with pytest.raises(ValueError):
        VerseRef("BLA", "1", "1")


def test_from_string_invalid_numbers() -> None:
    with pytest.raises(ValueError):
        VerseRef.from_string("EXO 6:-18")
    with pytest.raises(ValueError):
        VerseRef.from_string("EXO -1:18")


def test_from_string_letters() -> None:
    with pytest.raises(ValueError):
        VerseRef.from_string("EXO F:18")
    with pytest.raises(ValueError):
        VerseRef.from_string("EXO 1:F")


def test_copy_from() -> None:
    source = VerseRef("LUK", "3", "4b-6a", VULGATE_VERSIFICATION)
    dest = VerseRef()
    dest.copy_from(source)
    # Now change the source to ensure that we didn't just make it referentially equal.
    source.book_num = 2
    source.chapter_num = 6
    source.verse_num = 9
    source.versification = ENGLISH_VERSIFICATION

    assert dest.book == "LUK"
    assert dest.chapter_num == 3
    assert dest.verse == "4b-6a"
    assert dest.verse_num == 4
    assert dest.versification == VULGATE_VERSIFICATION


def test_copy_verse_from() -> None:
    source = VerseRef("LUK", "3", "4b-6a", VULGATE_VERSIFICATION)
    dest = VerseRef(1, 3, 5, RUSSIAN_ORTHODOX_VERSIFICATION)
    dest.copy_verse_from(source)
    # Now change the source to ensure that we didn't just make it referentially equal.
    source.book_num = 2
    source.chapter_num = 6
    source.verse_num = 9
    source.versification = ENGLISH_VERSIFICATION

    assert dest.book == "GEN"
    assert dest.chapter_num == 3
    assert dest.verse == "4b-6a"
    assert dest.verse_num == 4
    assert dest.versification == RUSSIAN_ORTHODOX_VERSIFICATION

    # Now test when the source just has a plain verse number (no bridges or segments)
    dest.copy_verse_from(source)
    assert dest.book == "GEN"
    assert dest.chapter_num == 3
    assert dest.verse == "9"
    assert dest.verse_num == 9
    assert dest.versification == RUSSIAN_ORTHODOX_VERSIFICATION


def test_all_verses_bridge() -> None:
    vref = VerseRef("LUK", "3", "4b-6a", VULGATE_VERSIFICATION)
    assert list(vref.all_verses()) == [
        VerseRef("LUK", "3", "4b", VULGATE_VERSIFICATION),
        VerseRef("LUK", "3", "5", VULGATE_VERSIFICATION),
        VerseRef("LUK", "3", "6a", VULGATE_VERSIFICATION),
    ]


def test_all_verses_simple_verse() -> None:
    vref = VerseRef("LUK", "3", "12", VULGATE_VERSIFICATION)
    assert list(vref.all_verses()) == [vref]


def test_all_verses_verse_with_segment() -> None:
    vref = VerseRef("LUK", "3", "12v", VULGATE_VERSIFICATION)
    assert list(vref.all_verses()) == [vref]


def test_get_ranges_single_verse() -> None:
    vref = VerseRef.from_string("LUK 3:12", ORIGINAL_VERSIFICATION)
    assert list(vref.get_ranges()) == [VerseRef.from_string("LUK 3:12", ORIGINAL_VERSIFICATION)]


def test_get_ranges_single_range() -> None:
    vref = VerseRef.from_string("LUK 3:12-14", ORIGINAL_VERSIFICATION)
    assert list(vref.get_ranges()) == [VerseRef.from_string("LUK 3:12-14", ORIGINAL_VERSIFICATION)]


def test_get_ranges_multiple_ranges() -> None:
    vref = VerseRef.from_string("LUK 3:12-14,16-17", ORIGINAL_VERSIFICATION)
    assert list(vref.get_ranges()) == [
        VerseRef.from_string("LUK 3:12-14", ORIGINAL_VERSIFICATION),
        VerseRef.from_string("LUK 3:16-17", ORIGINAL_VERSIFICATION),
    ]


def test_get_ranges_complicated_ranges() -> None:
    vref = VerseRef.from_string("LUK 3:12-14,16b-17a,18a,19,20", ORIGINAL_VERSIFICATION)
    assert list(vref.get_ranges()) == [
        VerseRef.from_string("LUK 3:12-14", ORIGINAL_VERSIFICATION),
        VerseRef.from_string("LUK 3:16b-17a", ORIGINAL_VERSIFICATION),
        VerseRef.from_string("LUK 3:18a", ORIGINAL_VERSIFICATION),
        VerseRef.from_string("LUK 3:19", ORIGINAL_VERSIFICATION),
        VerseRef.from_string("LUK 3:20", ORIGINAL_VERSIFICATION),
    ]


def test_lt() -> None:
    assert VerseRef(1, 1, 1) < VerseRef(2, 1, 1)
    assert not (VerseRef(10, 1, 1) < VerseRef(1, 1, 1))
    assert VerseRef("GEN", "1", "1a") < VerseRef("GEN", "1", "1b")
    assert VerseRef(1, 1, 1) < VerseRef("GEN", "1", "1a")
    assert not (VerseRef("GEN", "1", "1a") < VerseRef(1, 1, 1))


def test_le() -> None:
    assert VerseRef(1, 1, 1) <= VerseRef(2, 1, 1)
    assert not (VerseRef(10, 1, 1) <= VerseRef(1, 1, 1))
    assert VerseRef(1, 1, 1) <= VerseRef(1, 1, 1)
    assert VerseRef("GEN", "1", "1a") <= VerseRef("GEN", "1", "1b")
    assert VerseRef("GEN", "1", "1a") <= VerseRef("GEN", "1", "1a")
    assert VerseRef(1, 1, 1) <= VerseRef("GEN", "1", "1a")
    assert not (VerseRef("GEN", "1", "1a") <= VerseRef(1, 1, 1))


def test_gt() -> None:
    assert VerseRef(2, 1, 1) > VerseRef(1, 1, 1)
    assert not (VerseRef(1, 1, 1) > VerseRef(10, 1, 1))
    assert VerseRef("GEN", "1", "1b") > VerseRef("GEN", "1", "1a")
    assert VerseRef("GEN", "1", "1a") > VerseRef(1, 1, 1)
    assert not (VerseRef(1, 1, 1) > VerseRef("GEN", "1", "1a"))


def test_ge() -> None:
    assert VerseRef(2, 1, 1) >= VerseRef(1, 1, 1)
    assert not (VerseRef(1, 1, 1) >= VerseRef(10, 1, 1))
    assert VerseRef(1, 1, 1) >= VerseRef(1, 1, 1)
    assert VerseRef("GEN", "1", "1b") >= VerseRef("GEN", "1", "1a")
    assert VerseRef("GEN", "1", "1a") >= VerseRef("GEN", "1", "1a")
    assert VerseRef("GEN", "1", "1a") >= VerseRef(1, 1, 1)
    assert not (VerseRef(1, 1, 1) >= VerseRef("GEN", "1", "1a"))


def test_eq() -> None:
    assert VerseRef(1, 1, 1) == VerseRef(1, 1, 1)
    assert VerseRef("GEN", "1", "1a") == VerseRef("GEN", "1", "1a")
    assert VerseRef("GEN", "1", "1a") != VerseRef("GEN", "1", "1b")
    assert VerseRef("GEN", "1", "1a") != VerseRef(1, 1, 1)
    assert VerseRef("GEN", "1", "1a") != 1001001


def test_change_versification() -> None:
    vref = VerseRef.from_string("EXO 6:0", ENGLISH_VERSIFICATION)
    vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("EXO 6:0", ORIGINAL_VERSIFICATION)

    vref = VerseRef.from_string("GEN 31:55", ENGLISH_VERSIFICATION)
    vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("GEN 32:1", ORIGINAL_VERSIFICATION)

    vref = VerseRef.from_string("ESG 1:2", ENGLISH_VERSIFICATION)
    vref.change_versification(SEPTUAGINT_VERSIFICATION)
    assert vref == VerseRef.from_string("ESG 1:1b", SEPTUAGINT_VERSIFICATION)

    vref = VerseRef.from_string("ESG 1:1b", SEPTUAGINT_VERSIFICATION)
    vref.change_versification(ENGLISH_VERSIFICATION)
    assert vref == VerseRef.from_string("ESG 1:2", ENGLISH_VERSIFICATION)

    vref = VerseRef.from_string("ESG 1:3", RUSSIAN_ORTHODOX_VERSIFICATION)
    vref.change_versification(SEPTUAGINT_VERSIFICATION)
    assert vref == VerseRef.from_string("ESG 1:1c", SEPTUAGINT_VERSIFICATION)

    vref = VerseRef.from_string("ESG 1:1c", SEPTUAGINT_VERSIFICATION)
    vref.change_versification(RUSSIAN_ORTHODOX_VERSIFICATION)
    assert vref == VerseRef.from_string("ESG 1:3", RUSSIAN_ORTHODOX_VERSIFICATION)


def test_change_versification_with_ranges() -> None:
    vref = VerseRef.from_string("EXO 6:0", ENGLISH_VERSIFICATION)
    assert vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("EXO 6:0", ORIGINAL_VERSIFICATION)

    vref = VerseRef.from_string("GEN 31:55", ENGLISH_VERSIFICATION)
    assert vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("GEN 32:1", ORIGINAL_VERSIFICATION)

    vref = VerseRef.from_string("GEN 32:3-4", ENGLISH_VERSIFICATION)
    assert vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("GEN 32:4-5", ORIGINAL_VERSIFICATION)

    # This is the case where this can't really work properly
    vref = VerseRef.from_string("GEN 31:54-55", ENGLISH_VERSIFICATION)
    assert not vref.change_versification(ORIGINAL_VERSIFICATION)
    assert vref == VerseRef.from_string("GEN 31:54-1", ORIGINAL_VERSIFICATION)


def test_compare_to_with_without_verse_bridges() -> None:
    vref_without_bridge = VerseRef(1, 1, 2)
    vref_with_bridge = VerseRef.from_string("GEN 1:2-3")

    assert vref_with_bridge.compare_to(vref_without_bridge) > 0
    assert vref_without_bridge.compare_to(vref_with_bridge) < 0


def test_compare_to_same_verse_bridge() -> None:
    vref1 = VerseRef.from_string("GEN 1:1-2")
    vref2 = VerseRef.from_string("GEN 1:1-2")

    assert vref2.compare_to(vref1) == 0


def test_compare_to_overlapping_verse_bridges() -> None:
    vref1 = VerseRef.from_string("GEN 1:1-2")
    vref2 = VerseRef.from_string("GEN 1:2-3")

    assert vref2.compare_to(vref1) > 0
    assert vref1.compare_to(vref2) < 0


def test_compare_to_verse_lists() -> None:
    vref1 = VerseRef.from_string("GEN 1:2,3,21")
    vref2 = VerseRef.from_string("GEN 1:2,21")

    assert vref2.compare_to(vref1) > 0
    assert vref1.compare_to(vref2) < 0

    vref1 = VerseRef.from_string("GEN 1:2,3,21")
    vref2 = VerseRef.from_string("GEN 1:2,3")

    assert vref2.compare_to(vref1) < 0
    assert vref1.compare_to(vref2) > 0


def test_compare_to_verse_bridge_includes_another() -> None:
    vref1 = VerseRef.from_string("GEN 1:1-2")
    vref2 = VerseRef.from_string("GEN 1:1-5")

    assert vref2.compare_to(vref1) > 0
    assert vref1.compare_to(vref2) < 0


def test_compare_to_versification_makes_different_verse_same() -> None:
    vref1 = VerseRef.from_string("EXO 8:1", ENGLISH_VERSIFICATION)
    # Set up another VerseRef that has a different verse that is defined to be same as EXO 8:1 in the Septuagint
    # (The Septuagint is the same as original versification for these verses).
    vref2 = VerseRef.from_string("EXO 7:26", SEPTUAGINT_VERSIFICATION)

    assert vref2.compare_to(vref1) == 0
    assert vref1.compare_to(vref2) == 0


def test_compare_to_versification_makes_different_verse_range_same() -> None:
    vref1 = VerseRef.from_string("EXO 8:2-3", ENGLISH_VERSIFICATION)
    # Set up another VerseRef that has a different verse range that is defined to be same as EXO 8:2-3 in original
    # versification.
    vref2 = VerseRef.from_string("EXO 7:27-28", ORIGINAL_VERSIFICATION)

    assert vref2.compare_to(vref1) == 0
    assert vref1.compare_to(vref2) == 0


def test_compare_to_versification_makes_same_verse_different() -> None:
    vref1 = VerseRef.from_string("EXO 8:1", ENGLISH_VERSIFICATION)
    # Set up another VerseRef that has a different verse that is different from original.
    vref2 = VerseRef.from_string("EXO 8:1", ORIGINAL_VERSIFICATION)

    # Changing English ref to standard versification (EXO 8:1 => EXO 7:26) so difference (1) is found in chapter number
    # that is evaluated first.
    assert vref2.compare_to(vref1) > 0
    # Changing Septuagint ref to English versification EXO 8:1 => EXO 8:5 so difference (-4) is found in verse number
    # since book and chapter numbers are the same.
    assert vref1.compare_to(vref2) < 0


def test_compare_to_versification_makes_same_verse_range_different() -> None:
    vref1 = VerseRef.from_string("EXO 8:2-3", ENGLISH_VERSIFICATION)
    # Set up another VerseRef that has a different verse that is different from original.
    vref2 = VerseRef.from_string("EXO 8:2-3", SEPTUAGINT_VERSIFICATION)

    # Changing English ref to standard versification (EXO 8:2-3 => EXO 7:27-28) so difference (1) is found in chapter
    # number that is evaluated first.
    assert vref2.compare_to(vref1) > 0
    # Changing Septuagint ref to English versification (EXO 8:2-3 => EXO 8:6-7) so difference (-4) is found in verse
    # number since book and chapter numbers are the same.
    assert vref1.compare_to(vref2) < 0


def test_compare_to_segments() -> None:
    assert VerseRef.from_string("GEN 1:1a").compare_to(VerseRef.from_string("GEN 1:1")) > 0
    assert VerseRef.from_string("GEN 1:1").compare_to(VerseRef.from_string("GEN 1:1a")) < 0
    assert VerseRef.from_string("GEN 1:1a").compare_to(VerseRef.from_string("GEN 1:1b")) < 0
    assert VerseRef.from_string("GEN 1:1b").compare_to(VerseRef.from_string("GEN 1:1a")) > 0
    assert VerseRef.from_string("GEN 1:1a").compare_to(VerseRef.from_string("GEN 1:1a")) == 0
    assert VerseRef.from_string("GEN 1:1b").compare_to(VerseRef.from_string("GEN 1:1b")) == 0


def test_validated_segment() -> None:
    assert VerseRef.from_string("GEN 1:1").validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1a").validated_segment() == "a"
    assert VerseRef.from_string("GEN 1:1@").validated_segment() == "@"
    assert VerseRef.from_string("GEN 1:1a-5c").validated_segment() == "a"
    assert VerseRef.from_string("GEN 1:1-5c").validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1b-3c").validated_segment() == "b"
    assert VerseRef.from_string("GEN 1:1a,3,5").validated_segment() == "a"
    assert VerseRef.from_string("GEN 1:1,3b,5").validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1abc").validated_segment() == "abc"
    assert VerseRef.from_string("GEN 1:1a\u0301").validated_segment() == "a\u0301"


def test_validated_segment_with_versification_info() -> None:
    versification = Versification.create("Dummy")
    versification.verse_segments[get_bbbcccvvv(1, 1, 1)] = {"", "@", "$", "%", "abc", "a\u0301"}

    assert VerseRef.from_string("GEN 1:1", versification).validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1a", versification).validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1@", versification).validated_segment() == "@"
    assert VerseRef.from_string("GEN 1:1!", versification).validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1def", versification).validated_segment() == ""
    assert VerseRef.from_string("GEN 1:2a", versification).validated_segment() == "a"
    assert VerseRef.from_string("GEN 1:2b", versification).validated_segment() == "b"
    assert VerseRef.from_string("GEN 1:1abc", versification).validated_segment() == "abc"
    assert VerseRef.from_string("GEN 1:1abcdef", versification).validated_segment() == ""
    assert VerseRef.from_string("GEN 1:1a\u0301", versification).validated_segment() == "a\u0301"


def test_validated_segment_with_defined_default_segments() -> None:
    defined_segments = {"@", "$", "%", "abc", "a\u0301"}

    assert VerseRef.from_string("GEN 1:1").validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1a").validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1@").validated_segment(defined_segments) == "@"
    assert VerseRef.from_string("GEN 1:1$").validated_segment(defined_segments) == "$"
    assert VerseRef.from_string("GEN 1:1!").validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1abc").validated_segment(defined_segments) == "abc"
    assert VerseRef.from_string("GEN 1:1def").validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1a\u0301").validated_segment(defined_segments) == "a\u0301"


def test_validated_segment_with_versification_and_defined_default_segments() -> None:
    versification = Versification.create("Dummy")
    versification.verse_segments[get_bbbcccvvv(1, 1, 1)] = {"^", "&", "*", "a\u0301"}
    defined_segments = {"@", "$", "%", "o\u0301"}

    assert VerseRef.from_string("GEN 1:1*", versification).validated_segment(defined_segments) == "*"
    assert VerseRef.from_string("GEN 1:1a\u0301", versification).validated_segment(defined_segments) == "a\u0301"
    assert VerseRef.from_string("GEN 1:2a\u0301", versification).validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:2*", versification).validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1@", versification).validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:1o\u0301", versification).validated_segment(defined_segments) == ""
    assert VerseRef.from_string("GEN 1:2@", versification).validated_segment(defined_segments) == "@"
    assert VerseRef.from_string("GEN 1:2o\u0301", versification).validated_segment(defined_segments) == "o\u0301"


def test_str() -> None:
    assert str(VerseRef(1, 0, 0)) == "GEN 0:0"
    assert str(VerseRef(1, 1, 0)) == "GEN 1:0"
    assert str(VerseRef(1, 2, 0)) == "GEN 2:0"
    assert str(VerseRef(2, 4, 6)) == "EXO 4:6"
    assert str(VerseRef("LEV", "4", "6b-7a")) == "LEV 4:6b-7a"


def test_simplify() -> None:
    vref = VerseRef()
    vref.simplify()
    assert vref == VerseRef()

    vref = VerseRef.from_string("EXO 6:0")
    vref.simplify()
    assert vref == VerseRef.from_string("EXO 6:0")

    vref = VerseRef.from_string("EXO 6:5b-18a,19")
    vref.simplify()
    assert vref == VerseRef.from_string("EXO 6:5")

    vref = VerseRef.from_string("EXO 6:9a,9b")
    vref.simplify()
    assert vref == VerseRef.from_string("EXO 6:9")

    vref = VerseRef.from_string("EXO 6:4-10")
    vref.simplify()
    assert vref == VerseRef.from_string("EXO 6:4")

    vref = VerseRef.from_string("EXO 6:150monkeys")
    vref.simplify()
    assert vref == VerseRef.from_string("EXO 6:150")


def test_unbridge() -> None:
    assert VerseRef().unbridge() == VerseRef()
    assert VerseRef.from_string("EXO 6:0").unbridge() == VerseRef.from_string("EXO 6:0")
    assert VerseRef.from_string("EXO 6:5b-18a,19").unbridge() == VerseRef.from_string("EXO 6:5b")
    assert VerseRef.from_string("EXO 6:9a,9b").unbridge() == VerseRef.from_string("EXO 6:9a")
    assert VerseRef.from_string("EXO 6:4-10").unbridge() == VerseRef.from_string("EXO 6:4")
    assert VerseRef.from_string("EXO 6:150monkeys").unbridge() == VerseRef.from_string("EXO 6:150monkeys")
