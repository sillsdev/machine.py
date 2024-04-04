from typing import List, Optional, Tuple

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import parse_usfm
from machine.corpora.usfm_verse_text_updater import UsfmVerseTextUpdater
from machine.scripture import ENGLISH_VERSIFICATION, VerseRef


def test_get_usfm_char_style() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 1:1", ENGLISH_VERSIFICATION)],
            str("First verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 1 First verse of the first chapter.\r\n" in target


def test_get_usfm_id_text() -> None:
    target = update_usfm(id_text="- Updated")
    assert "\\id MAT - Updated\r\n" in target


def test_get_usfm_strip_all_text() -> None:
    target = update_usfm(strip_all_text=True)
    assert "\\id MAT\r\n" in target
    assert "\\v 1\r\n" in target
    assert "\\s\r\n" in target


def test_get_usfm_notes() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:1", ENGLISH_VERSIFICATION)],
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_row_verse_segment() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:1a", ENGLISH_VERSIFICATION)],
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_verse_segment() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:7", ENGLISH_VERSIFICATION)],
            str("Seventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 7a Seventh verse of the second chapter.\r\n" in target


def test_get_usfm_multiple_paras() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 1:2", ENGLISH_VERSIFICATION)],
            str("Second verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 2 Second verse of the first chapter.\r\n\\li2\r\n" in target


def test_get_usfm_table() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:9", ENGLISH_VERSIFICATION)],
            str("Ninth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 9 Ninth verse of the second chapter. \\tcr2 \\tc3 \\tcr4\r\n" in target


def test_get_usfm_range_single_row_multiple_verses() -> None:
    rows = [
        (
            [
                VerseRef.from_string("MAT 2:11", ENGLISH_VERSIFICATION),
                VerseRef.from_string("MAT 2:12", ENGLISH_VERSIFICATION),
            ],
            str("Eleventh verse of the second chapter. Twelfth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_range_single_row_single_verse() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:11", ENGLISH_VERSIFICATION)],
            str("Eleventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 11-12 Eleventh verse of the second chapter.\r\n" in target


def test_get_usfm_range_multiple_rows_single_verse() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:11", ENGLISH_VERSIFICATION)],
            str("Eleventh verse of the second chapter."),
        ),
        (
            [VerseRef.from_string("MAT 2:12", ENGLISH_VERSIFICATION)],
            str("Twelfth verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows)
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_opt_break() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:2", ENGLISH_VERSIFICATION)],
            str("Second verse of the second chapter."),
        ),
        (
            [VerseRef.from_string("MAT 2:3", ENGLISH_VERSIFICATION)],
            str("Third verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows)
    assert "\\v 2-3 Second verse of the second chapter. Third verse of the second chapter.\r\n" in target


def test_get_usfm_milestone() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 2:10", ENGLISH_VERSIFICATION)],
            str("Tenth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 10 Tenth verse of the second chapter. \\tc3-4 \\qt-s |Jesus\\*\\qt-e\\*\r\n" in target


def test_get_usfm_unmatched() -> None:
    rows = [
        (
            [VerseRef.from_string("MAT 1:3", ENGLISH_VERSIFICATION)],
            str("Third verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert "\\v 3 Third verse of the first chapter.\r\n" in target


def update_usfm(
    rows: Optional[List[Tuple[List[VerseRef], str]]] = None,
    id_text: Optional[str] = None,
    strip_all_text: bool = False,
) -> str:
    source = read_usfm()
    updater = UsfmVerseTextUpdater(rows, id_text, strip_all_text)
    parse_usfm(source, updater)
    return updater.get_usfm()


def read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
