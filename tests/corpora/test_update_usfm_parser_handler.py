from typing import Iterable, List, Optional, Sequence, Union

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, ignore_line_endings

from machine.corpora import (
    FileParatextProjectTextUpdater,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
    UsfmUpdateBlock,
    UsfmUpdateBlockElementType,
    UsfmUpdateBlockHandler,
    parse_usfm,
)


def test_get_usfm_verse_char_style() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("First verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert (
        "\\v 1 First verse of the first chapter. \\f + \\fr 1:1: \\ft This is a footnote for v1.\\f*\r\n\\li1\r\n\\v 2"
        in target
    )


def test_get_usfm_id_text() -> None:
    target = update_usfm(id_text="Updated")
    assert target is not None
    assert "\\id MAT - Updated\r\n" in target


def test_get_usfm_strip_all_text() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:3"),
            str("Update 3"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\r keep this reference
\rem and this reference too
\ip but remove this text
\v 1 Chapter \add one\add*, \p verse \f + \fr 1:1: \ft This is a \+bd ∆\+bd* footnote.\f*one.
\v 2 Chapter \add one\add*, \p verse \f + \fr 1:2: \ft This is a \+bd ∆\+bd* footnote.\f*two.
\v 3 Verse 3
\v 4 Verse 4
"""

    target = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior=UpdateUsfmMarkerBehavior.PRESERVE,
    )

    result = r"""\id MAT
    \c 1
    \r keep this reference
    \rem and this reference too
    \ip
    \v 1 Update 1 \add \add*
    \p \f + \fr 1:1: \ft This is a \+bd ∆\+bd* footnote.\f*
    \v 2 \add \add*
    \p \f + \fr 1:2: \ft This is a \+bd ∆\+bd* footnote.\f*
    \v 3 Update 3
    \v 4
    """
    assert_usfm_equals(target, result)

    target = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
        embed_behavior=UpdateUsfmMarkerBehavior.STRIP,
        style_behavior=UpdateUsfmMarkerBehavior.STRIP,
    )

    result = r"""\id MAT
\c 1
\r keep this reference
\rem and this reference too
\ip
\v 1 Update 1
\p
\v 2
\p
\v 3 Update 3
\v 4
"""
    assert_usfm_equals(target, result)


def test_get_usfm_strip_paragraphs_preserve_paragraph_styles():
    rows = [
        UpdateUsfmRow(scr_ref("MAT 1:0/1:rem"), "New remark"),
        UpdateUsfmRow(scr_ref("MAT 1:0/3:ip"), "Another new remark"),
        UpdateUsfmRow(scr_ref("MAT 1:1"), "Update 1"),
    ]
    usfm = r"""\id MAT
\c 1
\rem Update remark
\r reference
\ip This is another remark, but with a different marker
\v 1 This is a verse
"""

    target = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
    )
    result = r"""\id MAT
\c 1
\rem Update remark
\r reference
\ip Another new remark
\v 1 Update 1
"""

    assert_usfm_equals(target, result)

    target_diff_paragraph = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
        preserve_paragraph_styles=["ip"],
    )
    result_diff_paragraph = r"""\id MAT
\c 1
\rem New remark
\r
\ip This is another remark, but with a different marker
\v 1 Update 1
"""

    assert_usfm_equals(target_diff_paragraph, result_diff_paragraph)


def test_preserve_paragraphs():
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/1:rem"),
            str("Update remark"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT
\c 1
\rem Update remark
\r reference
\ip This is another remark, but with a different marker
\v 1 This is a verse
"""

    target = update_usfm(rows, usfm, text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING)
    result = r"""\id MAT
\c 1
\rem Update remark
\r reference
\ip
\v 1 Update 1
"""

    assert_usfm_equals(target, result)

    target_diff_paragraph = update_usfm(
        rows, usfm, text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING, preserve_paragraph_styles=("ip")
    )
    result_diff_paragraph = r"""\id MAT
\c 1
\rem Update remark
\r
\ip This is another remark, but with a different marker
\v 1 Update 1
"""

    assert_usfm_equals(target_diff_paragraph, result_diff_paragraph)


def test_paragraph_in_verse():
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p paragraph not in a verse
\v 1 verse 1 \p inner verse paragraph
\s1 Section Header
\v 2 Verse 2 \p inner verse paragraph
"""

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP)

    result = r"""\id MAT - Test
\c 1
\p paragraph not in a verse
\v 1 Update 1
\s1 Section Header
\v 2 Verse 2 inner verse paragraph
"""

    assert_usfm_equals(target, result)

    target_strip = update_usfm(
        rows,
        usfm,
        text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
        paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP,
    )

    result_strip = r"""\id MAT
\c 1
\p
\v 1 Update 1
\s1
\v 2
"""

    assert_usfm_equals(target_strip, result_strip)


def test_get_usfm_prefer_existing():
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:2"),
            str("Update 2"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Some text
\v 2
\v 3 Other text
"""

    target = update_usfm(rows, usfm, text_behavior=UpdateUsfmTextBehavior.PREFER_EXISTING)

    result = r"""\id MAT - Test
\c 1
\v 1 Some text
\v 2 Update 2
\v 3 Other text
"""
    assert_usfm_equals(target, result)


def test_get_usfm_prefer_rows():
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:6"),
            str("Text 6"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:7"),
            str("Text 7"),
        ),
    ]
    target = update_usfm(rows, text_behavior=UpdateUsfmTextBehavior.PREFER_NEW)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 6 Text 6\r\n" in target
    assert "\\v 7 Text 7\r\n" in target


def test_get_usfm_verse_strip_note() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:1"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    assert target is not None
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_verse_replace_with_note() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("updated text"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Chapter \add one\add*, verse \f + \fr 2:1: \ft This is a footnote.\f*one.
"""
    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 updated text \f + \fr 2:1: \ft This is a footnote.\f*
"""
    assert_usfm_equals(target, result)


def test_get_usfm_row_verse_segment() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:1a"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 1 First verse of the second chapter. \\f + \\fr 2:1: \\ft This is a footnote.\\f*\r\n" in target


def test_get_usfm_verse_segment() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:7"),
            str("Seventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 7a Seventh verse of the second chapter.\r\n" in target


def test_get_usfm_verse_multiple_paras() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:2"),
            str("Second verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert (
        "\\v 2 Second verse of the first chapter.\r\n\\li2 \\f + \\fr 1:2: \\ft This is a footnote for v2.\\f*"
        in target
    )


def test_get_usfm_verse_table() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:9"),
            str("Ninth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 9 Ninth verse of the second chapter. \\tcr2 \\tc3 \\tcr4\r\n" in target


def test_get_usfm_verse_range_single_row_multiple_verses() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:11", "MAT 2:12"),
            str("Eleventh verse of the second chapter. Twelfth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_verse_range_single_row_single_verse() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:11"),
            str("Eleventh verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter.\r\n" in target


def test_get_usfm_verse_range_multiple_rows_single_verse() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:11"),
            str("Eleventh verse of the second chapter."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:12"),
            str("Twelfth verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 11-12 Eleventh verse of the second chapter. Twelfth verse of the second chapter.\r\n" in target


def test_get_usfm_merge_verse_segments() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:2"),
            str("Verse 2."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:2a"),
            str("Verse 2a."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:2b"),
            str("Verse 2b."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 2-3 Verse 2. Verse 2a. Verse 2b.\r\n" in target


def test_get_usfm_verse_opt_break() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:2"),
            str("Second verse of the second chapter."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:3"),
            str("Third verse of the second chapter."),
        ),
    ]
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    assert target is not None
    assert "\\v 2-3 Second verse of the second chapter. Third verse of the second chapter.\r\n" in target


def test_get_usfm_verse_milestone() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:10"),
            str("Tenth verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 10 Tenth verse of the second chapter. \\tc3-4 \\qt-s |Jesus\\*\\qt-e\\*\r\n" in target


def test_get_usfm_verse_unmatched() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:3"),
            str("Third verse of the first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 3 Third verse of the first chapter.\r\n" in target


def test_get_usfm_nonverse_char_style() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:0/3:s1"),
            str("The second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s1 The second chapter.\r\n" in target


def test_get_usfm_nonverse_paragraph() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/8:s"),
            str("The first chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s The first chapter.\r\n" in target


def test_get_usfm_nonverse_relaxed() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/s"),
            str("The first chapter."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("First verse of the first chapter."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:0/tr/tc1"),
            str("The first cell of the table."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:0/tr/tc2"),
            str("The second cell of the table."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:0/tr/tc1"),
            str("The third cell of the table."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s The first chapter.\r\n" in target
    assert "\\v 1 First verse of the first chapter. \\f + \\fr 1:1: \\ft This is a footnote for v1.\\f*\r\n" in target
    assert "\\tr \\tc1 The first cell of the table. \\tc2 The second cell of the table.\r\n" in target
    assert "\\tr \\tc1 The third cell of the table. \\tc2 Row two, column two.\r\n" in target


def test_get_usfm_nonverse_sidebar() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:3/1:esb/1:ms"),
            str("The first paragraph of the sidebar."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\ms The first paragraph of the sidebar.\r\n" in target


def test_get_usfm_nonverse_table() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:0/1:tr/1:tc1"),
            str("The first cell of the table."),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 2:0/2:tr/1:tc1"),
            str("The third cell of the table."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\tr \\tc1 The first cell of the table. \\tc2 Row one, column two.\r\n" in target


def test_get_usfm_nonverse_optbreak() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:3/1:esb/2:p"),
            str("The second paragraph of the sidebar."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\p The second paragraph of the sidebar.\r\n" in target


def test_get_usfm_nonverse_milestone() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 2:7a/1:s"),
            str("A new section header."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\s A new section header. \\ts-s\\*\r\n" in target


def test_get_usfm_nonverse_skip_note() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        )
    ]
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    assert target is not None
    assert "\\ip The introductory paragraph.\r\n" in target


def test_get_usfm_nonverse_replace_with_note() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        ),
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\ip The introductory paragraph. \\fe + \\ft This is an endnote.\\fe*\r\n" in target


def test_get_usfm_verse_double_va_vp() -> None:
    rows = [
        UpdateUsfmRow(
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
        UpdateUsfmRow(
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
        UpdateUsfmRow(
            scr_ref("GEN 1:1"),
            str("Pretranslations before the start"),
        ),
        UpdateUsfmRow(
            scr_ref("GEN 1:2"),
            str("Pretranslations before the start"),
        ),
        UpdateUsfmRow(
            scr_ref("GEN 1:3"),
            str("Pretranslations before the start"),
        ),
        UpdateUsfmRow(
            scr_ref("GEN 1:4"),
            str("Pretranslations before the start"),
        ),
        UpdateUsfmRow(
            scr_ref("GEN 1:5"),
            str("Pretranslations before the start"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:0/3:ip"),
            str("The introductory paragraph."),
        ),
    ]

    target = update_usfm(rows)
    assert target is not None
    assert "\\ip The introductory paragraph. \\fe + \\ft This is an endnote.\\fe*\r\n" in target


def test_strip_paragraphs() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/2:p"),
            str("Update Paragraph"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update Verse 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p This is a second paragraph before any verses
\v 1 Hello
\q1 World
\p
\v 2 Hello
\p World
"""

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE)
    result_p = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p Update Paragraph
\v 1 Update Verse 1
\q1
\p
\v 2 Hello
\p World
"""

    assert_usfm_equals(target, result_p)

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result_s = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p Update Paragraph
\v 1 Update Verse 1
\p
\v 2 Hello World
"""
    assert_usfm_equals(target, result_s)


def test_preservation_raw_strings() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str(r"Update all in one row \f \fr 1.1 \ft Some note \f*"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 \f \fr 1.1 \ft Some note \f*Hello World
"""

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Update all in one row \f \fr 1.1 \ft Some note \f*
"""
    assert_usfm_equals(target, result)


def test_beginning_of_verse_embed() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str(r"Updated text"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 \f \fr 1.1 \ft Some note \f* Text after note
"""

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Updated text
"""
    assert_usfm_equals(target, result)


def test_cross_reference_dont_update() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1/1:x"),
            str("Update the cross reference"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Cross reference verse \x - \xo 2:3-4 \xt Cool Book 3:24 \xta The annotation \x* and more content.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Cross reference verse \x - \xo 2:3-4 \xt Cool Book 3:24 \xta The annotation \x* and more content.
"""
    assert_usfm_equals(target, result)


def test_preserve_fig() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \fig stuff\fig* more text and more.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update \fig stuff\fig*
"""
    assert_usfm_equals(target, result)


def test_note_explicit_end_markers() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update text"),
        ),
        UpdateUsfmRow(
            scr_ref("MAT 1:1/1:f"),
            str("Update note"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \f + \fr 2.4\fr* \fk The \+nd Lord\+nd*:\fk* \ft See \+nd Lord\+nd* in Word List.\ft*\f* and the end.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text \f + \fr 2.4\fr* \fk The \+nd Lord\+nd*:\fk* \ft See \+nd Lord\+nd* in Word List.\ft*\f*
"""
    assert_usfm_equals(target, result)

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text
"""
    assert_usfm_equals(target, result)


def test_update_block_verse_preserve_paras() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse 1 \p inner verse paragraph
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE, update_block_handlers=[update_block_handler]
    )

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse 1 ", True),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\p ", False),
        (UsfmUpdateBlockElementType.TEXT, "inner verse paragraph ", True),
    )


def test_update_block_verse_strip_paras() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse 1 \p inner verse paragraph
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(
        rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP, update_block_handlers=[update_block_handler]
    )

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse 1 ", True),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\p ", True),
        (UsfmUpdateBlockElementType.TEXT, "inner verse paragraph ", True),
    )


def test_update_block_verse_range() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1-3 verse 1 through 3
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE, update_block_handlers=[update_block_handler]
    )

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        ["MAT 1:1", "MAT 1:2", "MAT 1:3"],
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse 1 through 3 ", True),
    )


def test_update_block_footnote_preserve_embeds() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse\f \fr 1.1 \ft Some note \f* 1
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE, update_block_handlers=[update_block_handler]
    )

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse", True),
        (UsfmUpdateBlockElementType.EMBED, "\\f \\fr 1.1 \\ft Some note \\f*", False),
        (UsfmUpdateBlockElementType.TEXT, " 1 ", True),
    )


def test_update_block_footnote_strip_embeds() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse\f \fr 1.1 \ft Some note \f* 1
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse", True),
        (UsfmUpdateBlockElementType.EMBED, "\\f \\fr 1.1 \\ft Some note \\f*", True),
        (UsfmUpdateBlockElementType.TEXT, " 1 ", True),
    )


def test_update_block_nonverse() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:0/1:s"),
            str("Updated section Header"),
        ),
    ]
    usfm = r"""\id MAT - Test
\s Section header
\c 1
\v 1 verse 1
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 2
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:0/1:s",
        (UsfmUpdateBlockElementType.TEXT, "Updated section Header ", False),
        (UsfmUpdateBlockElementType.TEXT, "Section header ", True),
    )


def test_update_block_verse_preserve_styles() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse \bd 1\bd*
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(
        rows, usfm, style_behavior=UpdateUsfmMarkerBehavior.PRESERVE, update_block_handlers=[update_block_handler]
    )

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse ", True),
        (UsfmUpdateBlockElementType.STYLE, "\\bd ", False),
        (UsfmUpdateBlockElementType.TEXT, "1", True),
        (UsfmUpdateBlockElementType.STYLE, "\\bd*", False),
        (UsfmUpdateBlockElementType.TEXT, " ", True),
    )


def test_update_block_verse_strip_styles() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 verse \bd 1\bd*
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, style_behavior=UpdateUsfmMarkerBehavior.STRIP, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 1
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "verse ", True),
        (UsfmUpdateBlockElementType.STYLE, "\\bd ", True),
        (UsfmUpdateBlockElementType.TEXT, "1", True),
        (UsfmUpdateBlockElementType.STYLE, "\\bd*", True),
        (UsfmUpdateBlockElementType.TEXT, " ", True),
    )


def test_update_block_verse_section_header() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p
\v 1 Verse 1
\s Section header
\p
\v 2 Verse 2
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 4
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(update_block, "MAT 1:0/1:p")
    update_block = update_block_handler.blocks[1]
    assert_update_block_equals(update_block, "MAT 1:1/1:s", (UsfmUpdateBlockElementType.TEXT, "Section header ", False))
    update_block = update_block_handler.blocks[2]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "Verse 1 ", True),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\s Section header ", False),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\p ", False),
    )
    update_block = update_block_handler.blocks[3]
    assert_update_block_equals(
        update_block,
        "MAT 1:2",
        (UsfmUpdateBlockElementType.TEXT, "Verse 2 ", False),
    )


def test_update_block_verse_section_header_in_verse() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p
\v 1 Beginning of verse
\s Section header
\p end of verse
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 3
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(update_block, "MAT 1:0/1:p")
    update_block = update_block_handler.blocks[1]
    assert_update_block_equals(update_block, "MAT 1:1/1:s", (UsfmUpdateBlockElementType.TEXT, "Section header ", False))
    update_block = update_block_handler.blocks[2]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "Beginning of verse ", True),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\s Section header ", False),
        (UsfmUpdateBlockElementType.PARAGRAPH, "\\p ", False),
        (UsfmUpdateBlockElementType.TEXT, "end of verse ", True),
    )


def test_update_block_nonverse_paragraph_end_of_verse() -> None:
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p
\v 1 Verse 1
\s Section header
"""

    update_block_handler = _TestUsfmUpdateBlockHandler()
    update_usfm(rows, usfm, update_block_handlers=[update_block_handler])

    assert len(update_block_handler.blocks) == 3
    update_block = update_block_handler.blocks[0]
    assert_update_block_equals(update_block, "MAT 1:0/1:p")
    update_block = update_block_handler.blocks[1]
    assert_update_block_equals(update_block, "MAT 1:1/1:s", (UsfmUpdateBlockElementType.TEXT, "Section header ", False))
    update_block = update_block_handler.blocks[2]
    assert_update_block_equals(
        update_block,
        "MAT 1:1",
        (UsfmUpdateBlockElementType.TEXT, "Update 1 ", False),
        (UsfmUpdateBlockElementType.TEXT, "Verse 1 ", True),
    )


def test_header_reference_paragraphs() -> None:
    rows = [
        UpdateUsfmRow(scr_ref("MAT 1:1"), "new verse 1"),
        UpdateUsfmRow(scr_ref("MAT 1:2"), "new verse 2"),
        UpdateUsfmRow(scr_ref("MAT 1:3"), "new verse 3"),
        UpdateUsfmRow(scr_ref("MAT 2:1"), "new verse 1"),
        UpdateUsfmRow(scr_ref("MAT 2:2"), "new verse 2"),
    ]
    usfm = r"""\id MAT
\c 1
\s1 beginning-of-chapter header
\p
\v 1 verse 1
\s1 header between verses
\p
\v 2 verse 2
\s1 mid-verse header
\p more verse 2
\v 3 verse 3
\c 2
\v 1 consecutive elements
\s1 header
\r reference
\p
\v 2 verse 2
"""

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT
\c 1
\s1 beginning-of-chapter header
\p
\v 1 new verse 1
\s1 header between verses
\p
\v 2 new verse 2
\s1 mid-verse header
\p
\v 3 new verse 3
\c 2
\v 1 new verse 1
\s1 header
\r reference
\p
\v 2 new verse 2
"""
    assert_usfm_equals(target, result)


def test_pass_remark():
    rows = [
        UpdateUsfmRow(
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
    ]
    usfm = r"""\id MAT
\rem Existing remark
\c 1
\v 1 This is a verse
"""

    target = update_usfm(rows, usfm, text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING, remarks=["An added remark"])
    result = r"""\id MAT
\rem Existing remark
\rem An added remark
\c 1
\v 1 Update 1
"""

    assert_usfm_equals(target, result)

    target = update_usfm(
        rows, target, text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING, remarks=["Another added remark"]
    )
    result = r"""\id MAT
\rem Existing remark
\rem An added remark
\rem Another added remark
\c 1
\v 1 Update 1
"""

    assert_usfm_equals(target, result)


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def update_usfm(
    rows: Optional[Sequence[UpdateUsfmRow]] = None,
    source: Optional[str] = None,
    id_text: Optional[str] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_NEW,
    paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
    preserve_paragraph_styles: Optional[Iterable[str]] = None,
    update_block_handlers: Optional[Iterable[UsfmUpdateBlockHandler]] = None,
    remarks: Optional[Iterable[str]] = None,
) -> Optional[str]:
    if source is None:
        updater = FileParatextProjectTextUpdater(USFM_TEST_PROJECT_PATH)
        return updater.update_usfm(
            "MAT",
            rows,
            id_text,
            text_behavior,
            paragraph_behavior,
            embed_behavior,
            style_behavior,
            preserve_paragraph_styles,
            update_block_handlers,
            remarks,
        )
    else:
        source = source.strip().replace("\r\n", "\n") + "\r\n"
        updater = UpdateUsfmParserHandler(
            rows,
            id_text,
            text_behavior,
            paragraph_behavior,
            embed_behavior,
            style_behavior,
            preserve_paragraph_styles,
            update_block_handlers,
            remarks,
        )
        parse_usfm(source, updater)
        return updater.get_usfm()


def assert_usfm_equals(target: Optional[str], truth: str) -> None:
    assert target is not None
    for target_line, truth_line in zip(target.split("\n"), truth.split("\n")):
        assert target_line.strip() == truth_line.strip()


def read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()


def assert_update_block_equals(
    block: UsfmUpdateBlock,
    expected_ref: Union[str, Iterable[str]],
    *expected_elements: tuple[UsfmUpdateBlockElementType, str, bool],
) -> None:
    assert block.refs == [ScriptureRef.parse(expected_ref)] if isinstance(expected_ref, str) else list(expected_ref)
    assert len(block.elements) == len(expected_elements)
    for element, [expected_type, expected_usfm, expected_marked_for_removal] in zip(block.elements, expected_elements):
        assert element.type == expected_type
        assert "".join(token.to_usfm() for token in element.tokens) == expected_usfm
        assert element.marked_for_removal == expected_marked_for_removal


class _TestUsfmUpdateBlockHandler(UsfmUpdateBlockHandler):
    def __init__(self):
        self.blocks: list[UsfmUpdateBlock] = []

    def process_block(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock:
        new_block = block.copy()
        self.blocks.append(new_block)
        return new_block
