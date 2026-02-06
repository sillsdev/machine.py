from typing import List

from testutils.corpora_test_helpers import scripture_ref

from machine.corpora import ScriptureRef, TextRow, UsfmMemoryText
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


def test_get_sidebar_first_tag() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\esb
\ip My sidebar text
\esbe
\c 1
\p
\v 1 First verse
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 3, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == "My sidebar text"
    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:0/1:esb/1:ip")
    assert rows[1].text == ""
    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:0/2:p")
    assert rows[2].text == "First verse"
    assert scripture_ref(rows[2]) == ScriptureRef.parse("MAT 1:1")


def test_get_table_row_first_tag() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\tr \th1 Day \th2 Tribe \th3 Leader
\tr \tcr1 1st \tc2 Judah \tc3 Nahshon son of Amminadab
\c 1
\p
\v 1 First verse
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 8, str.join(",", [tr.text for tr in rows])
    assert rows[0].text == "\\th1 Day"
    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:0/1:tr/1:th1")
    assert rows[6].text == ""
    assert scripture_ref(rows[6]) == ScriptureRef.parse("MAT 1:0/3:p")
    assert rows[7].text == "First verse"
    assert scripture_ref(rows[7]) == ScriptureRef.parse("MAT 1:1")


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


def test_get_rows_paragraph_before_nonverse_paragraph() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v 1 verse 1
\b
\s1 header
\q1
\v 2 verse 2
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])
    assert rows[1].text == "verse 1 \\b \\q1"
    assert rows[2].text == "header"


def test_get_rows_verse_zero():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\h
\mt
\c 1
\p \v 0
\s
\p \v 1 Verse one.
"""
    )

    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0")
    assert rows[0].text == ""

    assert rows[1].ref == ScriptureRef.parse("MAT 1:1")
    assert rows[1].text == "Verse one."


def test_get_rows_style_starting_nonverse_paragraph_after_empty_paragraph() -> None:
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v 1 verse 1
\b
\s1 \w header\w*
\q1
\v 2 verse 2
""",
        include_all_text=True,
        include_markers=True,
    )
    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])
    assert rows[1].text == "verse 1 \\b \\q1"
    assert rows[2].text == "\\w header\\w*"


def test_get_rows_verse_zero_with_text():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\h
\mt
\c 1
\p \v 0 Verse zero.
\s
\p \v 1 Verse one.
"""
    )

    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0")
    assert rows[0].text == "Verse zero."

    assert rows[1].ref == ScriptureRef.parse("MAT 1:1")
    assert rows[1].text == "Verse one."


def test_get_rows_private_use_marker():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test English Apocrypha
\zmt Ignore this paragraph
\mt1 Test English Apocrypha
\pc Copyright Statement \zimagecopyrights
\pc Further copyright statements
""",
        include_all_text=True,
    )

    assert len(rows) == 3, str.join(",", [tr.text for tr in rows])

    assert rows[1].ref == ScriptureRef.parse("MAT 1:0/2:pc")
    assert rows[1].text == "Copyright Statement"


def test_get_rows_verse_range_with_right_to_left_marker():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\h
\mt
\c 1
\v 1"""
        + "\u200f"
        + r"""-2 Verse one and two.
"""
    )

    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:1")
    assert rows[0].text == "Verse one and two."

    assert rows[1].ref == ScriptureRef.parse("MAT 1:2")


def test_get_rows_non_latin_verse_number():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v १ Verse 1
\v 3,৪ Verses 3 and 4
\p
""",
        include_all_text=True,
    )

    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:p")
    assert rows[0].text == ""

    assert rows[1].ref == ScriptureRef.parse("MAT 1:1")
    assert rows[1].text == "Verse 1"

    assert rows[2].ref == ScriptureRef.parse("MAT 1:3")
    assert rows[2].text == "Verses 3 and 4"

    assert rows[3].ref == ScriptureRef.parse("MAT 1:৪")
    assert rows[3].text == ""


def test_get_rows_empty_verse_number():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v
\b
""",
        include_all_text=True,
    )

    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:p")
    assert rows[0].text == ""

    assert rows[1].ref == ScriptureRef.parse("MAT 1:0/2:b")
    assert rows[1].text == ""


def test_get_rows_multiple_empty_verse_numbers():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v
\p
\v
\p
\v
\p
""",
        include_all_text=True,
    )

    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])

    for i, row in enumerate(rows):
        assert row.ref == ScriptureRef.parse(f"MAT 1:0/{i+1}:p")
        assert row.text == ""


def test_get_rows_empty_verse_number_with_text():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\s heading text
\v  \vn 1 verse text
""",
        include_all_text=True,
    )

    assert len(rows) == 2, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:s")
    assert rows[0].text == "heading text"

    assert rows[1].ref == ScriptureRef.parse("MAT 1:0/2:vn")
    assert rows[1].text == "1 verse text"


def test_get_rows_empty_verse_number_mid_verse():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v 1 verse 1 text
\v
\v 2 verse 2 text
""",
        include_all_text=True,
    )

    assert len(rows) == 3, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:p")
    assert rows[0].text == ""

    assert rows[1].ref == ScriptureRef.parse("MAT 1:1")
    assert rows[1].text == "verse 1 text"

    assert rows[2].ref == ScriptureRef.parse("MAT 1:2")
    assert rows[2].text == "verse 2 text"


def test_get_rows_invalid_verse_numbers():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\p
\v BK1 text goes here
\v BK 2 text goes here
\v BK 3 text goes here
\v BK 4 text goes here
""",
        include_all_text=True,
    )

    assert len(rows) == 1, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:p")
    assert rows[0].text == "text goes here 2 text goes here 3 text goes here 4 text goes here"


def test_get_rows_incomplete_verse_range():
    rows: List[TextRow] = get_rows(
        r"""\id MAT - Test
\c 1
\s heading text
\p
\q1
\v 1,
\q1 verse 1 text
""",
        include_all_text=True,
    )

    assert len(rows) == 4, str.join(",", [tr.text for tr in rows])

    assert rows[0].ref == ScriptureRef.parse("MAT 1:0/1:s")
    assert rows[0].text == "heading text"

    assert rows[1].ref == ScriptureRef.parse("MAT 1:0/2:p")
    assert rows[1].text == ""

    assert rows[2].ref == ScriptureRef.parse("MAT 1:1/3:q1")
    assert rows[2].text == ""

    assert rows[3].ref == ScriptureRef.parse("MAT 1:1/4:q1")
    assert rows[3].text == "verse 1 text"


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
