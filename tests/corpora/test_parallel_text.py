from machine.corpora import (
    AlignedWordPair,
    MemoryText,
    MemoryTextAlignmentCollection,
    ParallelText,
    TextAlignment,
    TextSegment,
    TextSegmentRef,
)


def test_segments_no_segments() -> None:
    source_text = MemoryText("text1")
    target_text = MemoryText("text1")
    parallel_text = ParallelText(source_text, target_text)
    assert not any(parallel_text.segments)


def test_segments_no_missing_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 1), {AlignedWordPair(0, 0)}),
            TextAlignment("text1", TextSegmentRef(1, 2), {AlignedWordPair(1, 1)}),
            TextAlignment("text1", TextSegmentRef(1, 3), {AlignedWordPair(2, 2)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 3
    assert segments[0].source_segment == "source segment 1 1 .".split()
    assert segments[0].target_segment == "target segment 1 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[2].source_segment == "source segment 1 3 .".split()
    assert segments[2].target_segment == "target segment 1 3 .".split()
    assert segments[2].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_middle_target_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 1), {AlignedWordPair(0, 0)}),
            TextAlignment("text1", TextSegmentRef(1, 3), {AlignedWordPair(2, 2)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 .".split()
    assert segments[0].target_segment == "target segment 1 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment == "source segment 1 3 .".split()
    assert segments[1].target_segment == "target segment 1 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_middle_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 1), {AlignedWordPair(0, 0)}),
            TextAlignment("text1", TextSegmentRef(1, 3), {AlignedWordPair(2, 2)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 .".split()
    assert segments[0].target_segment == "target segment 1 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment == "source segment 1 3 .".split()
    assert segments[1].target_segment == "target segment 1 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_last_target_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 1), {AlignedWordPair(0, 0)}),
            TextAlignment("text1", TextSegmentRef(1, 2), {AlignedWordPair(1, 1)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 .".split()
    assert segments[0].target_segment == "target segment 1 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment == "source segment 1 2 .".split()
    assert segments[1].target_segment == "target segment 1 2 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_segments_missing_last_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 1), {AlignedWordPair(0, 0)}),
            TextAlignment("text1", TextSegmentRef(1, 2), {AlignedWordPair(1, 1)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 .".split()
    assert segments[0].target_segment == "target segment 1 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment == "source segment 1 2 .".split()
    assert segments[1].target_segment == "target segment 1 2 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_segments_missing_first_target_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 2), {AlignedWordPair(1, 1)}),
            TextAlignment("text1", TextSegmentRef(1, 3), {AlignedWordPair(2, 2)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 2 .".split()
    assert segments[0].target_segment == "target segment 1 2 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert segments[1].source_segment == "source segment 1 3 .".split()
    assert segments[1].target_segment == "target segment 1 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_first_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 2), "source segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            TextAlignment("text1", TextSegmentRef(1, 2), {AlignedWordPair(1, 1)}),
            TextAlignment("text1", TextSegmentRef(1, 3), {AlignedWordPair(2, 2)}),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 2 .".split()
    assert segments[0].target_segment == "target segment 1 2 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert segments[1].source_segment == "source segment 1 3 .".split()
    assert segments[1].target_segment == "target segment 1 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_range() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 2),
                "source segment 1 2 . source segment 1 3 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 3), in_range=True),
            TextSegment.create("text1", TextSegmentRef(1, 4), "source segment 1 4 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 4), "target segment 1 4 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.segments)
    assert len(segments) == 3
    assert segments[1].source_segment == "source segment 1 2 . source segment 1 3 .".split()
    assert segments[1].target_segment == "target segment 1 2 . target segment 1 3 .".split()


def test_segments_overlapping_ranges() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 2),
                "source segment 1 2 . source segment 1 3 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 3), in_range=True),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 1),
                "target segment 1 1 . target segment 1 2 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 2), in_range=True),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.segments)
    assert len(segments) == 1
    assert segments[0].source_segment == "source segment 1 1 . source segment 1 2 . source segment 1 3 .".split()
    assert segments[0].target_segment == "target segment 1 1 . target segment 1 2 . target segment 1 3 .".split()


def test_segments_adjacent_ranges_same_text() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 1),
                "source segment 1 1 . source segment 1 2 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 2), in_range=True),
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 3),
                "source segment 1 3 . source segment 1 4 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 4), in_range=True),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 4), "target segment 1 4 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 . source segment 1 2 .".split()
    assert segments[0].target_segment == "target segment 1 1 . target segment 1 2 .".split()
    assert segments[1].source_segment == "source segment 1 3 . source segment 1 4 .".split()
    assert segments[1].target_segment == "target segment 1 3 . target segment 1 4 .".split()


def test_segments_adjacent_ranges_different_texts() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 1),
                "source segment 1 1 . source segment 1 2 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 2), in_range=True),
            TextSegment.create("text1", TextSegmentRef(1, 3), "source segment 1 3 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 4), "source segment 1 4 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 3),
                "target segment 1 3 . target segment 1 4 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 4), in_range=True),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment == "source segment 1 1 . source segment 1 2 .".split()
    assert segments[0].target_segment == "target segment 1 1 . target segment 1 2 .".split()
    assert segments[1].source_segment == "source segment 1 3 . source segment 1 4 .".split()
    assert segments[1].target_segment == "target segment 1 3 . target segment 1 4 .".split()


def test_get_segments_range_all_target_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "source segment 1 1 .".split()),
            TextSegment.create(
                "text1",
                TextSegmentRef(1, 2),
                "source segment 1 2 . source segment 1 3 .".split(),
                in_range=True,
                range_start=True,
            ),
            TextSegment.create_empty("text1", TextSegmentRef(1, 3), in_range=True),
            TextSegment.create("text1", TextSegmentRef(1, 4), "source segment 1 4 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1, 1), "target segment 1 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 2), "target segment 1 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 3), "target segment 1 3 .".split()),
            TextSegment.create("text1", TextSegmentRef(1, 4), "target segment 1 4 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment == "source segment 1 2 . source segment 1 3 .".split()
    assert segments[1].is_source_in_range
    assert segments[1].is_source_range_start
    assert segments[1].target_segment == "target segment 1 2 .".split()
    assert not any(segments[2].source_segment)
    assert segments[2].is_source_in_range
    assert not segments[2].is_source_range_start
    assert segments[2].target_segment == "target segment 1 3 .".split()


def test_segments_same_ref_middle_many_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "source segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "source segment 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "target segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "target segment 3 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.segments)
    assert len(segments) == 6
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment == "source segment 2-1 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()
    assert segments[3].source_segment == "source segment 2-2 .".split()
    assert segments[3].target_segment == "target segment 2-1 .".split()
    assert segments[4].source_segment == "source segment 2-2 .".split()
    assert segments[4].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_one_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "source segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "source segment 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "target segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "target segment 3 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment == "source segment 2 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_many_to_one() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "source segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "source segment 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "target segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "target segment 3 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.get_segments(all_source_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[2].source_segment == "source segment 2-2 .".split()
    assert segments[2].target_segment == "target segment 2 .".split()


def test_get_segments_same_ref_last_one_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "source segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "target segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "target segment 3 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment == "source segment 2 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_last_many_to_one() -> None:
    source_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "source segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "source segment 2-2 .".split()),
            TextSegment.create("text1", TextSegmentRef(3), "source segment 3 .".split()),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            TextSegment.create("text1", TextSegmentRef(1), "target segment 1 .".split()),
            TextSegment.create("text1", TextSegmentRef(2), "target segment 2 .".split()),
        ],
    )

    parallel_text = ParallelText(source_text, target_text)
    segments = list(parallel_text.get_segments(all_source_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[2].source_segment == "source segment 2-2 .".split()
    assert segments[2].target_segment == "target segment 2 .".split()
