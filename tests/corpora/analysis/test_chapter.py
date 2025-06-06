from machine.corpora.analysis import Chapter, TextSegment, Verse


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

    assert len(chapter.get_verses()) == 2
    assert chapter.get_verses()[0].get_text_segments() == text_segments1
    assert chapter.get_verses()[1].get_text_segments() == text_segments2
