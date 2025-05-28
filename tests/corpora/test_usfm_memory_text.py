from typing import List, Optional, Sequence, Tuple

from testutils.corpora_test_helpers import scripture_ref

from machine.corpora import ScriptureRef, TextRow, UsfmMemoryText
from machine.corpora.update_usfm_parser_handler import (
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
)
from machine.corpora.usfm_parser import parse_usfm
from machine.corpora.usfm_stylesheet import UsfmStylesheet


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

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1"), str.join(",", [str(tr.ref) for tr in rows])
    assert rows[0].text == "Descriptive title", str.join(",", [tr.text for tr in rows])


def test_get_rows_last_segment() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\v 1 Last segment
"""
    )
    assert len(rows) == 1

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1"), str.join(",", [str(tr.ref) for tr in rows])
    assert rows[0].text == "Last segment", str.join(",", [tr.text for tr in rows])


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
    assert len(rows) == 5, str.join(",", [tr.text for tr in rows])


def test_get_rows_triplicate_verse() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\v 1 First verse 1
\rem non verse 1
\v 1 First verse 2
\rem non verse 2
\v 1 First verse 3
\rem non verse 3
\v 2 Second verse
""",
        include_all_text=True,
    )
    assert len(rows) == 5, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == "First verse 1"
    assert rows[3].text == "non verse 3"
    assert rows[4].text == "Second verse"


def test_get_rows_opt_break_middle_include_markers() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\v 1 First verse in line // More text
\c 2
\v 1
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == "First verse in line // More text"


def test_get_rows_verse_para_beginning_non_verse_segment() -> None:
    # a verse paragraph that begins with a non-verse segment followed by a verse segment
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\q1
\f \fr 119 \ft World \f*
\v 1 First verse in line!?!
\c 2
\d
description
\b
""",
        include_all_text=True,
    )
    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == ""
    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:0/1:q1")
    assert rows[1].text == "First verse in line!?!"
    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:1")


def test_get_rows_verse_para_comment_first() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\f \fr 119 \ft World \f*
\ip This is a comment
\c 1
\v 1 First verse in line!?!
\c 2
""",
        include_all_text=True,
    )
    assert rows[0].text == "This is a comment"
    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:0/2:ip")
    assert rows[1].text == "First verse in line!?!"
    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:1")
    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])


def test_get_rows_opt_break_outside_of_segment() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
//
\p
\v 1 This is the first verse.
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == ""
    assert rows[1].text == "This is the first verse."


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


def update_usfm(
    usfm: str,
    rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_EXISTING,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
) -> str:

    handler = UpdateUsfmParserHandler(rows, "MAT", text_behavior, embed_behavior, style_behavior)
    parse_usfm(usfm, handler)
    return handler.get_usfm()
