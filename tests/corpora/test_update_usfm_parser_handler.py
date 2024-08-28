from typing import List, Optional, Tuple

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, ignore_line_endings

from machine.corpora import ScriptureRef, parse_usfm
from machine.corpora.file_paratext_project_text_updater import FileParatextProjectTextUpdater
from machine.corpora.update_usfm_parser_handler import UpdateUsfmParserHandler


def test_get_usfm_verse_char_style() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("First verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 1 First verse of the first chapter.\r\n" in target


def test_get_usfm_id_text() -> None:
    target = update_usfm(id_text="Updated")
    assert target is not None
    assert "\\id MAT - Updated\r\n" in target


def test_get_usfm_strip_all_text() -> None:
    target = update_usfm(strip_all_text=True)
    assert target is not None
    assert "\\id MAT\r\n" in target
    assert "\\v 1\r\n" in target
    assert "\\s\r\n" in target


def test_get_usfm_prefer_existing():
    rows = [
        (
            scr_ref("MAT 1:6"),
            str("Text 6"),
        ),
        (
            scr_ref("MAT 1:7"),
            str("Text 7"),
        ),
    ]
    target = update_usfm(rows, prefer_existing_text=True)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 6 Verse 6 content.\r\n" in target
    assert "\\v 7 Text 7\r\n" in target


def test_get_usfm_prefer_rows():
    rows = [
        (
            scr_ref("MAT 1:6"),
            str("Text 6"),
        ),
        (
            scr_ref("MAT 1:7"),
            str("Text 7"),
        ),
    ]
    target = update_usfm(rows, prefer_existing_text=False)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 6 Text 6\r\n" in target
    assert "\\v 7 Text 7\r\n" in target


def test_get_usfm_verse_skip_note() -> None:
    rows = [
        (
            scr_ref("MAT 2:1"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_verse_replace_note() -> None:
    rows = [
        (
            scr_ref("MAT 2:1a"),
            str("First verse of the second chapter."),
        ),
        (scr_ref("MAT 2:1/1:f"), str("This is a new footnote.")),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 1 First verse of the second chapter. \\f + \\ft This is a new footnote.\\f*\r\n" in target


def test_get_usfm_row_verse_segment() -> None:
    rows = [
        (
            scr_ref("MAT 2:1a"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_verse_segment() -> None:
    rows = [
        (
            scr_ref("MAT 2:7"),
            str("Seventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 7a Seventh verse of the second chapter.\r\n" in target


def test_get_usfm_verse_multiple_paras() -> None:
    rows = [
        (
            scr_ref("MAT 1:2"),
            str("Second verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 2 Second verse of the first chapter.\r\n\\li2\r\n" in target


def test_get_usfm_verse_table() -> None:
    rows = [
        (
            scr_ref("MAT 2:9"),
            str("Ninth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 9 Ninth verse of the second chapter. \\tcr2 \\tc3 \\tcr4\r\n" in target


def test_get_usfm_verse_range_single_row_multiple_verses() -> None:
    rows = [
        (
            scr_ref("MAT 2:11", "MAT 2:12"),
            str("Eleventh verse of the second chapter. Twelfth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_verse_range_single_row_single_verse() -> None:
    rows = [
        (
            scr_ref("MAT 2:11"),
            str("Eleventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter.\r\n" in target


def test_get_usfm_verse_range_multiple_rows_single_verse() -> None:
    rows = [
        (
            scr_ref("MAT 2:11"),
            str("Eleventh verse of the second chapter."),
        ),
        (
            scr_ref("MAT 2:12"),
            str("Twelfth verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_merge_verse_segments() -> None:
    rows = [
        (
            scr_ref("MAT 2:2"),
            str("Verse 2."),
        ),
        (
            scr_ref("MAT 2:2a"),
            str("Verse 2a."),
        ),
        (
            scr_ref("MAT 2:2b"),
            str("Verse 2b."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 2-3 Verse 2. Verse 2a. Verse 2b.\r\n" in target


def test_get_usfm_verse_opt_break() -> None:
    rows = [
        (
            scr_ref("MAT 2:2"),
            str("Second verse of the second chapter."),
        ),
        (
            scr_ref("MAT 2:3"),
            str("Third verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 2-3 Second verse of the second chapter. Third verse of the second chapter.\r\n" in target


def test_get_usfm_verse_milestone() -> None:
    rows = [
        (
            scr_ref("MAT 2:10"),
            str("Tenth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 10 Tenth verse of the second chapter. \\tc3-4 \\qt-s |Jesus\\*\\qt-e\\*\r\n" in target


def test_get_usfm_verse_unmatched() -> None:
    rows = [
        (
            scr_ref("MAT 1:3"),
            str("Third verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 3 Third verse of the first chapter.\r\n" in target


def test_get_usfm_nonverse_char_style() -> None:
    rows = [
        (
            scr_ref("MAT 2:0/3:s1"),
            str("The second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s1 The second chapter.\r\n" in target


def test_get_usfm_nonverse_paragraph() -> None:
    rows = [
        (
            scr_ref("MAT 1:0/8:s"),
            str("The first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s The first chapter.\r\n" in target


def test_get_usfm_nonverse_relaxed() -> None:
    rows = [
        (
            scr_ref("MAT 1:0/s"),
            str("The first chapter."),
        ),
        (
            scr_ref("MAT 1:1"),
            str("First verse of the first chapter."),
        ),
        (
            scr_ref("MAT 2:0/tr/tc1"),
            str("The first cell of the table."),
        ),
        (
            scr_ref("MAT 2:0/tr/tc2"),
            str("The second cell of the table."),
        ),
        (
            scr_ref("MAT 2:0/tr/tc1"),
            str("The third cell of the table."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s The first chapter.\r\n" in target
    assert "\\v 1 First verse of the first chapter.\r\n" in target
    assert "\\tr \\tc1 The first cell of the table. \\tc2 The second cell of the table.\r\n" in target
    assert "\\tr \\tc1 The third cell of the table. \\tc2 Row two, column two.\r\n" in target


def test_get_usfm_nonverse_sidebar() -> None:
    rows = [
        (
            scr_ref("MAT 2:3/1:esb/1:ms"),
            str("The first paragraph of the sidebar."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\ms The first paragraph of the sidebar.\r\n" in target


def test_get_usfm_nonverse_table() -> None:
    rows = [
        (
            scr_ref("MAT 2:0/1:tr/1:tc1"),
            str("The first cell of the table."),
        ),
        (
            scr_ref("MAT 2:0/2:tr/1:tc1"),
            str("The third cell of the table."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\tr \\tc1 The first cell of the table. \\tc2 Row one, column two.\r\n" in target


def test_get_usfm_nonverse_optbreak() -> None:
    rows = [
        (
            scr_ref("MAT 2:3/1:esb/2:p"),
            str("The second paragraph of the sidebar."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\p The second paragraph of the sidebar.\r\n" in target


def test_get_usfm_nonverse_milestone() -> None:
    rows = [
        (
            scr_ref("MAT 2:7a/1:s"),
            str("A new section header."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s A new section header. \\ts-s\\*\r\n" in target


def test_get_usfm_nonverse_skip_note() -> None:
    rows = [
        (
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\ip The introductory paragraph.\r\n" in target


def test_get_usfm_nonverse_replace_note() -> None:
    rows = [
        (
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        ),
        (
            scr_ref("MAT 1:0/3:ip/1:fe"),
            str("This is a new endnote."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\ip The introductory paragraph. \\fe + \\ft This is a new endnote.\\fe*\r\n" in target


def test_get_usfm_verse_double_va_vp() -> None:
    rows = [
        (
            scr_ref("MAT 3:1"),
            str("Updating later in the book to start."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 1 \\va 2\\va*\\vp 1 (2)\\vp*Updating later in the book to start.\r\n" in target


def test_get_usfm_verse_last_segment() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Updating the last verse."),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1
"""
    target = update_usfm(rows, usfm)
    assert target is not None
    ignore_line_endings(
        target,
        r"""\id MAT - Test
\c 1
\v 1 Updating the last verse.
""",
    )


def test_get_usfm_verse_pretranslations_before_text() -> None:
    rows = [
        (
            scr_ref("GEN 1:1"),
            str("Pretranslations before the start"),
        ),
        (
            scr_ref("GEN 1:2"),
            str("Pretranslations before the start"),
        ),
        (
            scr_ref("GEN 1:3"),
            str("Pretranslations before the start"),
        ),
        (
            scr_ref("GEN 1:4"),
            str("Pretranslations before the start"),
        ),
        (
            scr_ref("GEN 1:5"),
            str("Pretranslations before the start"),
        ),
        (
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        ),
    ]

    target = update_usfm(rows)
    assert target is not None
    assert "\\ip The introductory paragraph.\r\n" in target


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def update_usfm(
    rows: Optional[List[Tuple[List[ScriptureRef], str]]] = None,
    source: Optional[str] = None,
    id_text: Optional[str] = None,
    strip_all_text: bool = False,
    prefer_existing_text: bool = False,
) -> Optional[str]:
    if source is None:
        updater = FileParatextProjectTextUpdater(USFM_TEST_PROJECT_PATH)
        return updater.update_usfm("MAT", rows, id_text, strip_all_text, prefer_existing_text)
    else:
        source = source.strip().replace("\r\n", "\n") + "\r\n"
        updater = UpdateUsfmParserHandler(rows, id_text, strip_all_text, prefer_existing_text)
        parse_usfm(source, updater)
        return updater.get_usfm()


def read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
