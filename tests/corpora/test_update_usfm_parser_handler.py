from typing import List, Optional, Sequence, Tuple

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, ignore_line_endings

from machine.corpora import (
    FileParatextProjectTextUpdater,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    parse_usfm,
)


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
        (
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
        (
            scr_ref("MAT 1:3"),
            str("Update 3"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\r keep this reference
\rem and this reference too
\ip but remove this text
\v 1 Chapter \add one\add*, \p verse \f + \fr 2:1: \ft This is a \fm ∆\fm* footnote.\f*one.
\v 2 Chapter \add one\add*, \p verse \f + \fr 2:1: \ft This is a \fm ∆\fm* footnote.\f*two.
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
\p \f + \fr 2:1: \ft \fm ∆\fm*\f*
\v 2 \add \add*
\p \f + \fr 2:1: \ft \fm ∆\fm*\f*
\v 3 Update 3
\v 4
"""
    assess(target, result)

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
\v 2
\v 3 Update 3
\v 4
"""
    assess(target, result)


def test_get_usfm_strip_paragraphs_preserve_paragraph_styles():
    rows = [
        (scr_ref("MAT 1:0/1:rem"), "New remark"),
        (scr_ref("MAT 1:0/3:ip"), "Another new remark"),
        (scr_ref("MAT 1:1"), "Update 1"),
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

    assess(target, result)

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

    assess(target_diff_paragraph, result_diff_paragraph)


def test_preserve_paragraphs():
    rows = [
        (
            scr_ref("MAT 1:0/1:rem"),
            str("Update remark"),
        ),
        (
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

    assess(target, result)

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

    assess(target_diff_paragraph, result_diff_paragraph)


def test_paragraph_in_verse():
    rows = [
        (
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
\v 2 Verse 2
\p inner verse paragraph
"""

    assess(target, result)

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

    assess(target_strip, result_strip)


def test_get_usfm_prefer_existing():
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update 1"),
        ),
        (
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
    assess(target, result)


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
    target = update_usfm(rows, text_behavior=UpdateUsfmTextBehavior.PREFER_NEW)
    assert target is not None
    assert "\\id MAT - Test\r\n" in target
    assert "\\v 6 Text 6\r\n" in target
    assert "\\v 7 Text 7\r\n" in target


def test_get_usfm_verse_strip_note() -> None:
    rows = [
        (
            scr_ref("MAT 2:1"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    assert target is not None
    assert "\\v 1 First verse of the second chapter.\r\n" in target


def test_get_usfm_verse_replace_note() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("updated text"),
        ),
        (scr_ref("MAT 1:1/1:f"), str("This is a new footnote.")),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Chapter \add one\add*, verse \f + \fr 2:1: \ft This is a \fq quotation \ft and an \fqa alternative quotation\f*one.
"""
    target = update_usfm(rows, usfm)
    # Only the first \ft marker is updated
    result = r"""\id MAT - Test
\c 1
\v 1 updated text \f + \fr 2:1: \ft This is a new footnote. \fq quotation \ft and an \fqa alternative quotation\f*
"""
    assess(target, result)


def test_get_usfm_row_verse_segment() -> None:
    rows = [
        (
            scr_ref("MAT 2:1a"),
            str("First verse of the second chapter."),
        )
    ]
    target = update_usfm(rows)
    assert target is not None
    assert "\\v 1 First verse of the second chapter. \\f + \\fr 2:1: \\ft This is a footnote.\\f*\r\n" in target


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
    assert (
        "\\v 2 Second verse of the first chapter.\r\n\\li2 \\f + \\fr 1:2: \\ft This is a footnote for v2.\\f*"
        in target
    )


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
    assert "\\v 2-3 Verse 2. Verse 2a. Verse 2b. \\fm ∆\\fm*\r\n" in target


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
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
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
    assert "\\v 1 First verse of the first chapter. \\f + \\fr 1:1: \\ft This is a footnote for v1.\\f*\r\n" in target
    assert "\\tr \\tc1 The first cell of the table. \\tc2 The second cell of the table.\r\n" in target
    assert "\\tr \\tc1 The third cell of the table. \\tc2 Row two, column two.\r\n" in target


def test_get_usfm_nonverse_sidebar() -> None:
    rows = [
        (
            scr_ref("MAT 2:3/2:esb/1:ms"),
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
            scr_ref("MAT 2:3/2:esb/2:p"),
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
    target = update_usfm(rows, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
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
    assert "\\ip The introductory paragraph. \\fe + \\ft This is a new endnote. \\fe*\r\n" in target


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
    assert "\\ip The introductory paragraph. \\fe + \\ft This is an endnote.\\fe*\r\n" in target


def test_embed_style_preservation() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update the greeting"),
        ),
        (
            scr_ref("MAT 1:1/1:f"),
            str("Update the comment"),
        ),
        (
            scr_ref("MAT 1:2"),
            str("Update the greeting only"),
        ),
        (
            scr_ref("MAT 1:3/1:f"),
            str("Update the comment only"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Hello \f \fr 1.1 \ft Some \+bd note\+bd* \f*\bd World \bd*
\v 2 Good \f \fr 1.2 \ft Some other \+bd note\+bd* \f*\bd Morning \bd*
\v 3 Pleasant \f \fr 1.3 \ft A third \+bd note\+bd* \f*\bd Evening \bd*
"""

    target = update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE, style_behavior=UpdateUsfmMarkerBehavior.PRESERVE
    )
    result_pp = r"""\id MAT - Test
\c 1
\v 1 Update the greeting \f \fr 1.1 \ft Update the comment \+bd \+bd*\f*\bd \bd*
\v 2 Update the greeting only \f \fr 1.2 \ft Some other \+bd note\+bd* \f*\bd \bd*
\v 3 Pleasant \f \fr 1.3 \ft Update the comment only \+bd \+bd*\f*\bd Evening \bd*
"""
    assess(target, result_pp)

    target = update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.PRESERVE, style_behavior=UpdateUsfmMarkerBehavior.STRIP
    )
    result_ps = r"""\id MAT - Test
\c 1
\v 1 Update the greeting \f \fr 1.1 \ft Update the comment \f*
\v 2 Update the greeting only \f \fr 1.2 \ft Some other \+bd note\+bd* \f*
\v 3 Pleasant \f \fr 1.3 \ft Update the comment only \f*\bd Evening \bd*
"""
    assess(target, result_ps)

    target = update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP, style_behavior=UpdateUsfmMarkerBehavior.PRESERVE
    )
    result_sp = r"""\id MAT - Test
\c 1
\v 1 Update the greeting \bd \bd*
\v 2 Update the greeting only \bd \bd*
\v 3 Pleasant \bd Evening \bd*
"""
    assess(target, result_sp)

    target = update_usfm(
        rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP, style_behavior=UpdateUsfmMarkerBehavior.STRIP
    )
    result_ss = r"""\id MAT - Test
\c 1
\v 1 Update the greeting
\v 2 Update the greeting only
\v 3 Pleasant \bd Evening \bd*
"""
    assess(target, result_ss)


def test_strip_paragraphs() -> None:
    rows = [
        (
            scr_ref("MAT 1:0/2:p"),
            str("Update Paragraph"),
        ),
        (
            scr_ref("MAT 1:1"),
            str("Update Verse 1"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p This is a second paragraph before any verses
\v 1 Hello
\p World
\v 2 Hello
\p World
"""

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.PRESERVE)
    result_p = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p Update Paragraph
\v 1 Update Verse 1
\p
\v 2 Hello
\p World
"""

    assess(target, result_p)

    target = update_usfm(rows, usfm, paragraph_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result_s = r"""\id MAT - Test
\c 1
\p This is a paragraph before any verses
\p Update Paragraph
\v 1 Update Verse 1
\v 2 Hello
\p World
"""
    assess(target, result_s)


def test_preservation_raw_strings() -> None:
    rows = [
        (
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
    assess(target, result)


def test_beginning_of_verse_embed() -> None:
    rows = [
        (
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
    assess(target, result)


def test_empty_note() -> None:
    rows = [
        (
            scr_ref("MAT 1:1/1:f"),
            str("Update the note"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Empty Note \f \fr 1.1 \ft \f*
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Empty Note \f \fr 1.1 \ft Update the note \f*
"""
    assess(target, result)


def test_cross_reference_dont_update() -> None:
    rows = [
        (
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
    assess(target, result)


def test_preserve_fig_and_fm() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \fig stuff\fig* more text \fm * \fm* and more.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update \fig stuff\fig*\fm * \fm*
"""
    assess(target, result)


def test_nested_xt() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update text"),
        ),
        (
            scr_ref("MAT 1:1/1:f"),
            str("Update note"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \f + \fr 15.8 \ft Text (\+xt reference\+xt*). And more.\f* and the end.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text \f + \fr 15.8 \ft Update note \+xt reference\+xt*\f*
"""
    assess(target, result)

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text
"""
    assess(target, result)


def test_non_nested_xt() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update text"),
        ),
        (
            scr_ref("MAT 1:1/1:f"),
            str("Update note"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \f + \fr 15.8 \ft Text \xt reference\f* and the end.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text \f + \fr 15.8 \ft Update note \xt reference\f*
"""
    assess(target, result)

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text
"""
    assess(target, result)


def test_multiple_ft_only_update_first() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update text"),
        ),
        (
            scr_ref("MAT 1:1/1:f"),
            str("Update note"),
        ),
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 initial text \f + \fr 15.8 \ft first note \ft second note\f* and the end.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text \f + \fr 15.8 \ft Update note \ft second note\f*
"""
    assess(target, result)

    target = update_usfm(rows, usfm, embed_behavior=UpdateUsfmMarkerBehavior.STRIP)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text
"""
    assess(target, result)


def test_implicitly_closed_char_style() -> None:
    rows = [
        (
            scr_ref("MAT 1:1"),
            str("Update text"),
        )
    ]
    usfm = r"""\id MAT - Test
\c 1
\v 1 Verse \bd one.
\c 2
\v 1 Verse one.
"""

    target = update_usfm(rows, usfm)
    result = r"""\id MAT - Test
\c 1
\v 1 Update text
\c 2
\v 1 Verse one.
"""
    assess(target, result)


def scr_ref(*refs: str) -> List[ScriptureRef]:
    return [ScriptureRef.parse(ref) for ref in refs]


def update_usfm(
    rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
    source: Optional[str] = None,
    id_text: Optional[str] = None,
    text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_NEW,
    paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
    style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
    preserve_paragraph_styles: Optional[Sequence[str]] = None,
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
        )
    else:
        source = source.strip().replace("\r\n", "\n") + "\r\n"
        updater = UpdateUsfmParserHandler(
            rows, id_text, text_behavior, paragraph_behavior, embed_behavior, style_behavior, preserve_paragraph_styles
        )
        parse_usfm(source, updater)
        return updater.get_usfm()


def assess(target: Optional[str], truth: str) -> None:
    assert target is not None
    for target_line, truth_line in zip(target.split("\n"), truth.split("\n")):
        assert target_line.strip() == truth_line.strip()


def read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
