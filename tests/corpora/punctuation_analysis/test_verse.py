from machine.corpora.punctuation_analysis import TextSegment, Verse


def test_initialize_verse() -> None:
    text_segments = [
        TextSegment.Builder().set_text("Segment 1").build(),
        TextSegment.Builder().set_text("Segment 2").build(),
        TextSegment.Builder().set_text("Segment 3").build(),
    ]

    verse = Verse(text_segments)

    assert len(verse.text_segments) == 3
    assert verse.text_segments == text_segments


def test_segment_indices() -> None:
    text_segments = [
        TextSegment.Builder().set_text("Segment 1").build(),
        TextSegment.Builder().set_text("Segment 1").build(),
        TextSegment.Builder().set_text("Segment 1").build(),
    ]

    verse = Verse(text_segments)

    assert verse.text_segments[0]._index_in_verse == 0
    assert verse.text_segments[1]._index_in_verse == 1
    assert verse.text_segments[2]._index_in_verse == 2


def test_num_segments_in_verse() -> None:
    text_segments = [
        TextSegment.Builder().set_text("Segment 1").build(),
        TextSegment.Builder().set_text("Segment 2").build(),
        TextSegment.Builder().set_text("Segment 3").build(),
    ]

    verse = Verse(text_segments)

    assert verse.text_segments[0]._num_segments_in_verse == 3
    assert verse.text_segments[1]._num_segments_in_verse == 3
    assert verse.text_segments[2]._num_segments_in_verse == 3
