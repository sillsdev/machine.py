from typing import List

from machine.corpora import UsfmParser
from machine.corpora.punctuation_analysis import Chapter, TextSegment, UsfmMarkerType, UsfmStructureExtractor, Verse

verse_text_parser_state = usfm_parser = UsfmParser("").state
verse_text_parser_state.verse_ref.verse_num = 1


def test_chapter_and_verse_markers():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_start_paragraph_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.start_para(verse_text_parser_state, "p", False, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.PARAGRAPH)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_start_character_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.start_char(verse_text_parser_state, "k", False, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.CHARACTER)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_end_character_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.end_char(verse_text_parser_state, "k", None, False)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.CHARACTER)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_end_note_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.end_note(verse_text_parser_state, "f", False)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.EMBED)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_end_table_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.end_note(verse_text_parser_state, "tr", False)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.EMBED)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_ref_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.end_note(verse_text_parser_state, "x", False)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.EMBED)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_sidebar_marker():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.end_note(verse_text_parser_state, "esb", False)
    usfm_structure_extractor.text(verse_text_parser_state, "test")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.EMBED)
                        .build()
                    ]
                )
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None


def test_multiple_verses():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")
    usfm_structure_extractor.verse(verse_text_parser_state, "2", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test2")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build()
                    ]
                ),
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test2")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build()
                    ]
                ),
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None
    assert actual_chapters[0].verses[1]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[1]._text_segments[0].next_segment is None


def test_multiple_chapters():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")
    usfm_structure_extractor.chapter(verse_text_parser_state, "2", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test2")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build()
                    ]
                ),
            ]
        ),
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test2")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build()
                    ]
                ),
            ]
        ),
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert actual_chapters[0].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[0].verses[0]._text_segments[0].next_segment is None
    assert actual_chapters[1].verses[0]._text_segments[0].previous_segment is None
    assert actual_chapters[1].verses[0]._text_segments[0].next_segment is None


def test_character_marker_in_text():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")
    usfm_structure_extractor.start_char(verse_text_parser_state, "k", False, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test2")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build(),
                        TextSegment.Builder()
                        .set_text("test2")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.CHARACTER)
                        .build(),
                    ]
                ),
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert (
        actual_chapters[0].verses[0]._text_segments[1].previous_segment
        == expected_chapters[0].verses[0]._text_segments[0]
    )
    assert (
        actual_chapters[0].verses[0]._text_segments[0].next_segment == expected_chapters[0].verses[0]._text_segments[1]
    )


def test_empty_text():
    usfm_structure_extractor = UsfmStructureExtractor()
    usfm_structure_extractor.chapter(verse_text_parser_state, "1", "c", None, None)
    usfm_structure_extractor.verse(verse_text_parser_state, "1", "v", None, None)
    usfm_structure_extractor.text(verse_text_parser_state, "test")
    usfm_structure_extractor.start_char(verse_text_parser_state, "k", False, None)
    usfm_structure_extractor.text(verse_text_parser_state, "")
    usfm_structure_extractor.end_char(verse_text_parser_state, "k", None, False)
    usfm_structure_extractor.text(verse_text_parser_state, "test2")

    expected_chapters = [
        Chapter(
            [
                Verse(
                    [
                        TextSegment.Builder()
                        .set_text("test")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .build(),
                        TextSegment.Builder()
                        .set_text("test2")
                        .add_preceding_marker(UsfmMarkerType.CHAPTER)
                        .add_preceding_marker(UsfmMarkerType.VERSE)
                        .add_preceding_marker(UsfmMarkerType.CHARACTER)
                        .build(),
                    ]
                ),
            ]
        )
    ]

    actual_chapters = usfm_structure_extractor.get_chapters()
    assert_chapter_equal(expected_chapters, actual_chapters)
    assert (
        actual_chapters[0].verses[0]._text_segments[1].previous_segment
        == expected_chapters[0].verses[0]._text_segments[0]
    )
    assert (
        actual_chapters[0].verses[0]._text_segments[0].next_segment == expected_chapters[0].verses[0]._text_segments[1]
    )


def assert_chapter_equal(expected_chapters: List[Chapter], actual_chapters: List[Chapter]):
    assert len(expected_chapters) == len(actual_chapters)
    for expected_chapter, actual_chapter in zip(expected_chapters, actual_chapters):
        assert len(expected_chapter.verses) == len(actual_chapter.verses)
        for expected_verse, actual_verse in zip(expected_chapter.verses, actual_chapter.verses):
            assert len(expected_verse._text_segments) == len(actual_verse._text_segments)
            for expected_segment, actual_segment in zip(expected_verse._text_segments, actual_verse._text_segments):
                assert expected_segment == actual_segment
