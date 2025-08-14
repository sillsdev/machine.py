from machine.punctuation_analysis import Chapter, TextSegment, Verse


def test_initialize_verse() -> None:
    text_segments1 = [
        TextSegment.Builder().set_text("Segment 1").build(),
        TextSegment.Builder().set_text("Segment 2").build(),
        TextSegment.Builder().set_text("Segment 3").build(),
    ]
    verse1 = Verse(text_segments1)

    text_segments2 = [
        TextSegment.Builder().set_text("Segment 4").build(),
        TextSegment.Builder().set_text("Segment 5").build(),
        TextSegment.Builder().set_text("Segment 6").build(),
    ]
    verse2 = Verse(text_segments2)

    chapter = Chapter([verse1, verse2])

    assert len(chapter.verses) == 2
    assert chapter.verses[0].text_segments == text_segments1
    assert chapter.verses[1].text_segments == text_segments2
