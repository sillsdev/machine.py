from testutils.memory_paratext_project_file_handler import DefaultParatextProjectSettings

from machine.corpora import ConvertUsfmVersificationHandler, UsfmTokenizer, parse_usfm
from machine.scripture import (
    ENGLISH_VERSIFICATION,
    ORIGINAL_VERSIFICATION,
    RUSSIAN_ORTHODOX_VERSIFICATION,
    Versification,
)


def test_one_fewer_chapter() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\c 4
\p
\s1 Section
\v 1-5
\v 6 Text
"""
    target = convert_usfm(usfm, ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\p
\s1 Section
\v 19-23
\v 24 Text
"""
    assert_usfm_equals(target, result)


def test_one_more_chapter() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\v 19-23
\v 24 Text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\c 4
\nb
\v 1-5
\v 6 Text
"""
    assert_usfm_equals(target, result)


def test_one_fewer_book() -> None:
    # Russian Orthodox vs. Original
    # PSA 151:1-7 = PS2 1:1-7
    usfm = r"""\id PSA - Test
\h Psalms
\c 150
\p
\v 1-5 Lines
\v 6 Line
\q Another line
\c 151
\p
\v 1-7 More lines
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id PSA - Test
\h Psalms
\c 150
\p
\v 1-5 Lines
\v 6 Line
\q Another line
"""
    assert_usfm_equals(target, result)


def test_one_more_book() -> None:
    # Russian Orthodox vs. Original
    # DAN 3:24-90 = DAG 3:24-90
    # DAN 3:91-100 = DAN 3:24-33
    usfm = r"""\id DAN - Test
\h Daniel
\c 3
\p
\v 1-23 Text 1
\v 24-90 Text 2
\p More text 2
\v 91-100 Text 3
\c 4
\p
\v 1 Text 4
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\h Daniel
\c 3
\p
\v 1-23 Text 1
\v 24-33 Text 3
\c 4
\p
\v 1 Text 4
"""
    assert_usfm_equals(target, result)


def test_back_one_verse_to_previous_chapter() -> None:
    # English vs. Original
    # ISA 9:1 = ISA 8:23
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22
\v 23
\c 9
\p
\v 1
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22
\c 9
\nb
\v 1
\p
\v 2
"""
    assert_usfm_equals(target, result)


def test_forward_one_verse_to_next_chapter() -> None:
    # Original vs. English
    # ISA 8:23 = ISA 9:1
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22
\c 9
\p
\v 1
\v 2
"""
    target = convert_usfm(usfm, ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22
\p
\v 23
\c 9
\nb
\v 1
"""
    assert_usfm_equals(target, result)


def test_cross_chapter_verse_range() -> None:
    # English vs. Original
    # ISA 9:1 = ISA 8:23
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22-23
\c 9
\p
\v 1
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22
\c 9
\nb
\v 1
\p
\v 2
"""
    assert_usfm_equals(target, result)


def test_cross_chapter_verse_range_cross_book() -> None:
    # Russian Orthodox vs. Original
    # DAN 3:24-90 = DAG 3:24-90
    # DAN 3:91-100 = DAN 3:24-33
    usfm = r"""\id DAN - Test
\c 3
\p
\v 1-22
\v 23-89
\v 90-100
\c 4
\p
\v 1
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\c 3
\p
\v 1-22
\v 23
\v 24-33
\c 4
\p
\v 1
"""
    assert_usfm_equals(target, result)


def test_cross_chapter_verse_range_cross_book_within_single_range() -> None:
    # Russian Orthodox vs. Original
    # DAN 3:24-90 = DAG 3:24-90
    # DAN 3:91-100 = DAN 3:24-33
    usfm = r"""\id DAN - Test
\c 3
\p
\v 1-100
\c 4
\p
\v 1
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\c 3
\p
\v 1-33
\c 4
\p
\v 1
"""
    assert_usfm_equals(target, result)


def test_heading_introducing_kept_verse_is_preserved() -> None:
    # Russian Orthodox vs. Original
    # DAN 3:24-90 = DAG 3:24-90
    # DAN 3:91-100 = DAN 3:24-33
    usfm = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\v 24-90 Dropped text
\s1 \nd Section\nd*
\p
\v 91-100 More text
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\s1 \nd Section\nd*
\p
\v 24-33 More text
"""
    assert_usfm_equals(target, result)


def test_heading_introducing_dropped_verse_is_dropped() -> None:
    # Russian Orthodox vs. Original
    # PSA 151:1-7 = PS2 1:1-7
    usfm = r"""\id PSA - Test
\c 150
\p
\v 1-5 Lines
\v 6 Line
\q Another line
\c 151
\s1 \nd Section\nd*
\p
\v 1-7 More lines
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id PSA - Test
\c 150
\p
\v 1-5 Lines
\v 6 Line
\q Another line
"""
    assert_usfm_equals(target, result)


def test_paragraph_introducing_dropped_verse_is_dropped() -> None:
    # Russian Orthodox vs. Original
    # DAN 3:24-90 = DAG 3:24-90
    # DAN 3:91-100 = DAN 3:24-33
    usfm = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\v 24-50 Dropped text
\q1
\v 51-90 More dropped text
\p
\v 91-100 More text
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\p
\v 24-33 More text
"""
    assert_usfm_equals(target, result)


def test_drop_verse_text() -> None:
    usfm = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\v 24-90 Dropped text
\v 91-100 More text
"""
    target = convert_usfm(usfm, RUSSIAN_ORTHODOX_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id DAN - Test
\c 3
\p
\v 1-23 Text
\v 24-33 More text
"""
    assert_usfm_equals(target, result)


def test_chapter_marker_is_followed_by_paragraph_marker() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL - Test
\c 3
\p
\v 1-18 Text
\v 19-23 More text
\v 24 Last text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id MAL - Test
\c 3
\p
\v 1-18 Text
\c 4
\nb
\v 1-5 More text
\v 6 Last text
"""
    assert_usfm_equals(target, result)


def test_chapter_marker_is_followed_by_paragraph_marker_cross_chapter_verse_range() -> None:
    # English vs. Original
    # ISA 9:1 = ISA 8:23
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22-23
\c 9
\p
\v 1
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22
\c 9
\nb
\v 1
\p
\v 2
"""
    assert_usfm_equals(target, result)


def test_chapter_marker_is_followed_by_paragraph_marker_heading_opens_paragraph() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL - Test
\c 3
\p
\v 18 Text
\s1 Section
\p
\v 19-23 More text
\v 24 Last text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id MAL - Test
\c 3
\p
\v 18 Text
\c 4
\s1 Section
\p
\v 1-5 More text
\v 6 Last text
"""
    assert_usfm_equals(target, result)


def test_heading_after_chapter_label_keeps_marker_content() -> None:
    # English vs. Original
    # ISA 9:1 = ISA 8:23
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22 Text
\c 9
\cl Chapter Nine
\s1 Section
\p
\v 1 Nine one
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22 Text
\c 9
\cl Chapter Nine
\s1 Section
\p
\v 2 Nine one
"""
    assert_usfm_equals(target, result)


def test_cross_chapter_verse_range_text_stays_with_first_verse() -> None:
    # English vs. Original
    # ISA 9:1 = ISA 8:23
    usfm = r"""\id ISA - Test
\c 8
\p
\v 22-23 Verse twenty-two and twenty-three text
\c 9
\p
\v 1 Chapter nine verse one text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id ISA - Test
\c 8
\p
\v 22 Verse twenty-two and twenty-three text
\c 9
\nb
\v 1
\p
\v 2 Chapter nine verse one text
"""
    assert_usfm_equals(target, result)


def test_ignore_invalid_chapter() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2@
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\v 19-23
\v 24 Text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)

    # Strip out invalid chapters since we can't reliably convert them
    result = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\c 4
\nb
\v 1-5
\v 6 Text
"""
    assert_usfm_equals(target, result)


def test_ignore_invalid_verse() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1@ Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\v 19-23
\v 24 Text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)

    # Just pass invalid verses through to target
    result = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1@ Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\c 4
\nb
\v 1-5
\v 6 Text
"""
    assert_usfm_equals(target, result)


def test_missing_verse_in_range() -> None:
    # English vs. Original
    # MAL 4:1-6 = MAL 3:19-24
    usfm = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\v 19-21,23 Text
\v 24 Text
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id MAL
\h Malachi
\c 1
\s1 Section
\p
\v 1 Text
\v 2-14
\c 2
\v 1-17
\c 3
\p
\v 1-17
\v 18 Text \f More text \f*
\c 4
\nb
\v 1-3 Text
\v 5
\v 6 Text
"""
    assert_usfm_equals(target, result)


def test_same_source_and_target_versification() -> None:
    usfm = r"""\id MAT - Test
\h Matthew
\mt Matthew
\ip An introduction to Matthew\fe + \ft This is an endnote.\fe*
\p \rq MAT 1\rq* Here is another paragraph.
\p and with a \w keyword|a special concept\w* in it.
\p and a \weirdtaglookingthing that is not an actual tag.
\c 1
\s Chapter One
\v 1 Chapter \pn one\+pro WON\+pro*\pn*, verse one.\f + \fr 1:1: \ft This is a footnote for v1.\f*
\li1
\v 2 \bd C\bd*hapter one,
\li2 verse\f + \fr 1:2: \ft This is a footnote for v2.\f* two.
\v 3 Chapter one \w*,
\li2 verse three.
\v 4 Chapter one with odd whitespace,
\li2 verse four,
\v 5 Chapter one,
\li2 verse \fig Figure 1|src="image1.png" size="col" ref="1:5"\fig* five.
\v 6 Verse 6 content.
\v 7
\v 8
"""
    target = convert_usfm(usfm, ENGLISH_VERSIFICATION, ENGLISH_VERSIFICATION)
    assert_usfm_equals(target, usfm)


def test_preceding_headings_not_moved() -> None:
    # English vs. Original
    # JOL 2:27-28 = JOL 2:27-3:1
    usfm = r"""\id JOL
\c 2
\v 27 Then you will know that I am present in Israel
\q2 and that I am the LORD your God,
\q2 and there is no other.
\q1 My people will never again
\q2 be put to shame.
\s1 I Will Pour Out My Spirit
\r (Acts 2:14–36)
\q1
\v 28 And afterward, I will pour out My Spirit on all people.
\q2 Your sons and daughters will prophesy,
\q1 your old men will dream dreams,
\q2 your young men will see visions.
"""
    target = convert_usfm(usfm, ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION)
    result = r"""\id JOL
\c 2
\v 27 Then you will know that I am present in Israel
\q2 and that I am the LORD your God,
\q2 and there is no other.
\q1 My people will never again
\q2 be put to shame.
\c 3
\s1 I Will Pour Out My Spirit
\r (Acts 2:14–36)
\q1
\v 1 And afterward, I will pour out My Spirit on all people.
\q2 Your sons and daughters will prophesy,
\q1 your old men will dream dreams,
\q2 your young men will see visions.
"""
    assert_usfm_equals(target, result)


def test_merged_verses() -> None:
    # Original vs. English
    # PSA 51:1-3 = PSA 51:0-1
    usfm = r"""\id PSA
\c 51
\s1 Create in Me a Clean Heart, O God
\r (2 Samuel 12:1–12)
\p
\v 1 For the choirmaster. A Psalm of David.
\v 2 When Nathan the prophet came to him after his adultery with Bathsheba.
\b
\q1
\v 3 Have mercy on me, O God,
\q2 according to Your loving devotion;
\q1 according to Your great compassion,
\q2 blot out my transgressions.
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, ENGLISH_VERSIFICATION)
    result = r"""\id PSA
\c 51
\s1 Create in Me a Clean Heart, O God
\r (2 Samuel 12:1–12)
\p
\v 0 For the choirmaster. A Psalm of David. When Nathan the prophet came to him after his adultery with Bathsheba.
\b
\q1
\v 1 Have mercy on me, O God,
\q2 according to Your loving devotion;
\q1 according to Your great compassion,
\q2 blot out my transgressions.
"""
    assert_usfm_equals(target, result)


def test_merged_verses_range_extends_past_merged_verse() -> None:
    # Original vs. Russian Orthodox
    # LEV 14:55-56 = LEV 14:55
    usfm = r"""\id LEV
\c 14
\p
\v 55 for leprosy in a garment or in a house,
\v 56-57 for a swelling, a rash, or a spot, to determine when something is clean or unclean.
"""
    target = convert_usfm(usfm, ORIGINAL_VERSIFICATION, RUSSIAN_ORTHODOX_VERSIFICATION)
    result = r"""\id LEV
\c 14
\p
\v 55 for leprosy in a garment or in a house,
\v 56 for a swelling, a rash, or a spot, to determine when something is clean or unclean.
"""
    assert_usfm_equals(target, result)


def convert_usfm(source: str, source_versification: Versification, target_versification: Versification) -> str:
    source = source.strip().replace("\r\n", "\n") + "\r\n"
    settings = DefaultParatextProjectSettings(
        versification=source_versification, file_name_form="MAT", file_name_suffix=""
    )
    handler = ConvertUsfmVersificationHandler(target_versification)
    tokenizer = UsfmTokenizer(settings.stylesheet)
    tokens = tokenizer.tokenize(source)
    parse_usfm(tokens, handler, settings.stylesheet, settings.versification)
    return handler.get_usfm(settings.stylesheet)


def assert_usfm_equals(target: str, truth: str) -> None:
    assert target is not None
    target_lines = target.split("\n")
    truth_lines = truth.split("\n")
    for i, truth_line in enumerate(truth_lines):
        assert target_lines[i].strip() == truth_line.strip(), f"Line {i}"
