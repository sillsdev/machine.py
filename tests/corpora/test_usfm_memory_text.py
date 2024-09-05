from typing import List

from testutils.corpora_test_helpers import scripture_ref

from machine.corpora import ScriptureRef, TextRow, UsfmMemoryText, UsfmStylesheet


def test_get_rows_verse_descriptive_title() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\d
\v 1 Descriptive title
\c 2
\b
\q1
\s
"""
    )
    assert len(rows) == 1

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1")
    assert rows[0].text == "Descriptive title"


def test_get_rows_last_segment() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\v 1 Last segment
"""
    )
    assert len(rows) == 1

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1")
    assert rows[0].text == "Last segment"


def test_get_rows_duplicate_verse_with_table() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\v 1 First verse
\periph Table of Contents Abbreviation
\rem non verse content 1
\v 1 duplicate first verse
\rem non verse content 2
\mt1 Table
\tr \tc1 row 1 cell 1 \tc2 row 1 cell 2
\tr \tc1 row 2 cell 1 \tc2 row 2 cell 2
""",
        include_all_text=True,
    )
    assert len(rows) == 5


def get_rows(usfm: str, include_markers: bool = False, include_all_text: bool = False) -> List[TextRow]:
    text = UsfmMemoryText(
        UsfmStylesheet("usfm.sty"),
        "utf-8",
        "MAT",
        usfm.strip().replace("\r\n", "\n").replace("\r", "\n") + "\r\n",
        include_markers=include_markers,
        include_all_text=include_all_text,
    )

    return list(text.get_rows())
