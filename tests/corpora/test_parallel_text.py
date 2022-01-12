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
    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    assert not any(parallel_text.segments)


def test_segments_no_missing_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 .", is_sentence_start=False),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 .", is_sentence_start=False),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(1, AlignedWordPair(0, 0)),
            alignment(2, AlignedWordPair(1, 1)),
            alignment(3, AlignedWordPair(2, 2)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 3
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 .".split()
    assert segments[0].target_segment == "target segment 1 .".split()
    assert not segments[0].is_source_sentence_start
    assert segments[0].is_target_sentence_start
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[2].source_segment_ref == TextSegmentRef(3)
    assert segments[2].target_segment_ref == TextSegmentRef(3)
    assert segments[2].source_segment == "source segment 3 .".split()
    assert segments[2].target_segment == "target segment 3 .".split()
    assert segments[2].is_source_sentence_start
    assert not segments[2].is_target_sentence_start
    assert segments[2].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_middle_target_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(3, "target segment 3 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(1, AlignedWordPair(0, 0)),
            alignment(3, AlignedWordPair(2, 2)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 .".split()
    assert segments[0].target_segment == "target segment 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 .".split()
    assert segments[1].target_segment == "target segment 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_middle_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(1, AlignedWordPair(0, 0)),
            alignment(3, AlignedWordPair(2, 2)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 .".split()
    assert segments[0].target_segment == "target segment 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 .".split()
    assert segments[1].target_segment == "target segment 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_last_target_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(1, AlignedWordPair(0, 0)),
            alignment(2, AlignedWordPair(1, 1)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 .".split()
    assert segments[0].target_segment == "target segment 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_segments_missing_last_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(1, AlignedWordPair(0, 0)),
            alignment(2, AlignedWordPair(1, 1)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 .".split()
    assert segments[0].target_segment == "target segment 1 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_segments_missing_first_target_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(2, AlignedWordPair(1, 1)),
            alignment(3, AlignedWordPair(2, 2)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(2)
    assert segments[0].target_segment_ref == TextSegmentRef(2)
    assert segments[0].source_segment == "source segment 2 .".split()
    assert segments[0].target_segment == "target segment 2 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 .".split()
    assert segments[1].target_segment == "target segment 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_missing_first_source_segment() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
        ],
    )
    alignments = MemoryTextAlignmentCollection(
        "text1",
        [
            alignment(2, AlignedWordPair(1, 1)),
            alignment(3, AlignedWordPair(2, 2)),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, alignments)
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(2)
    assert segments[0].target_segment_ref == TextSegmentRef(2)
    assert segments[0].source_segment == "source segment 2 .".split()
    assert segments[0].target_segment == "target segment 2 .".split()
    assert segments[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 .".split()
    assert segments[1].target_segment == "target segment 3 .".split()
    assert segments[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_segments_range() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(
                2,
                "source segment 2 . source segment 3 .",
                is_sentence_start=False,
                is_in_range=True,
                is_range_start=True,
            ),
            segment(3, is_in_range=True),
            segment(4, "source segment 4 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
            segment(4, "target segment 4 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.segments)
    assert len(segments) == 3
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 . source segment 3 .".split()
    assert segments[1].target_segment == "target segment 2 . target segment 3 .".split()
    assert not segments[1].is_source_sentence_start
    assert segments[1].is_target_sentence_start


def test_segments_overlapping_ranges() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 . source segment 3 .", is_in_range=True, is_range_start=True),
            segment(3, is_in_range=True),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 . target segment 2 .", is_in_range=True, is_range_start=True),
            segment(2, is_in_range=True),
            segment(3, "target segment 3 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.segments)
    assert len(segments) == 1
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 . source segment 2 . source segment 3 .".split()
    assert segments[0].target_segment == "target segment 1 . target segment 2 . target segment 3 .".split()
    assert segments[0].is_source_sentence_start
    assert segments[0].is_target_sentence_start


def test_segments_adjacent_ranges_same_text() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(
                1,
                "source segment 1 . source segment 2 .",
                is_sentence_start=False,
                is_in_range=True,
                is_range_start=True,
            ),
            segment(2, is_in_range=True),
            segment(3, "source segment 3 . source segment 4 .", is_in_range=True, is_range_start=True),
            segment(4, is_in_range=True),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 .", is_sentence_start=False),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
            segment(4, "target segment 4 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 . source segment 2 .".split()
    assert segments[0].target_segment == "target segment 1 . target segment 2 .".split()
    assert not segments[0].is_source_sentence_start
    assert not segments[0].is_target_sentence_start
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 . source segment 4 .".split()
    assert segments[1].target_segment == "target segment 3 . target segment 4 .".split()
    assert segments[1].is_source_sentence_start
    assert segments[1].is_target_sentence_start


def test_segments_adjacent_ranges_different_texts() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 . source segment 2 .", is_in_range=True, is_range_start=True),
            segment(2, is_in_range=True),
            segment(3, "source segment 3 ."),
            segment(4, "source segment 4 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 . target segment 4 .", is_in_range=True, is_range_start=True),
            segment(4, is_in_range=True),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.segments)
    assert len(segments) == 2
    assert segments[0].source_segment_ref == TextSegmentRef(1)
    assert segments[0].target_segment_ref == TextSegmentRef(1)
    assert segments[0].source_segment == "source segment 1 . source segment 2 .".split()
    assert segments[0].target_segment == "target segment 1 . target segment 2 .".split()
    assert segments[1].source_segment_ref == TextSegmentRef(3)
    assert segments[1].target_segment_ref == TextSegmentRef(3)
    assert segments[1].source_segment == "source segment 3 . source segment 4 .".split()
    assert segments[1].target_segment == "target segment 3 . target segment 4 .".split()


def test_get_segments_all_source_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
            segment(4, "source segment 4 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(3, "target segment 3 ."),
            segment(4, "target segment 4 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_source_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref is None
    assert segments[1].source_segment == "source segment 2 .".split()
    assert not any(segments[1].target_segment)


def test_get_segments_range_all_target_segments() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 . source segment 3 .", is_in_range=True, is_range_start=True),
            segment(3, is_in_range=True),
            segment(4, "source segment 4 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
            segment(4, "target segment 4 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 . source segment 3 .".split()
    assert segments[1].is_source_in_range
    assert segments[1].is_source_range_start
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(3)
    assert segments[2].target_segment_ref == TextSegmentRef(3)
    assert not any(segments[2].source_segment)
    assert segments[2].is_source_in_range
    assert not segments[2].is_source_range_start
    assert segments[2].target_segment == "target segment 3 .".split()


def test_segments_same_ref_middle_many_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2-1 ."),
            segment(2, "source segment 2-2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2-1 ."),
            segment(2, "target segment 2-2 ."),
            segment(3, "target segment 3 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.segments)
    assert len(segments) == 6
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(2)
    assert segments[2].target_segment_ref == TextSegmentRef(2)
    assert segments[2].source_segment == "source segment 2-1 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()
    assert segments[3].source_segment_ref == TextSegmentRef(2)
    assert segments[3].target_segment_ref == TextSegmentRef(2)
    assert segments[3].source_segment == "source segment 2-2 .".split()
    assert segments[3].target_segment == "target segment 2-1 .".split()
    assert segments[4].source_segment_ref == TextSegmentRef(2)
    assert segments[4].target_segment_ref == TextSegmentRef(2)
    assert segments[4].source_segment == "source segment 2-2 .".split()
    assert segments[4].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_one_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2-1 ."),
            segment(2, "target segment 2-2 ."),
            segment(3, "target segment 3 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(2)
    assert segments[2].target_segment_ref == TextSegmentRef(2)
    assert segments[2].source_segment == "source segment 2 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_many_to_one() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2-1 ."),
            segment(2, "source segment 2-2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
            segment(3, "target segment 3 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_source_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(2)
    assert segments[2].target_segment_ref == TextSegmentRef(2)
    assert segments[2].source_segment == "source segment 2-2 .".split()
    assert segments[2].target_segment == "target segment 2 .".split()


def test_get_segments_same_ref_last_one_to_many() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2-1 ."),
            segment(2, "target segment 2-2 ."),
            segment(3, "target segment 3 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_target_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2 .".split()
    assert segments[1].target_segment == "target segment 2-1 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(2)
    assert segments[2].target_segment_ref == TextSegmentRef(2)
    assert segments[2].source_segment == "source segment 2 .".split()
    assert segments[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_last_many_to_one() -> None:
    source_text = MemoryText(
        "text1",
        [
            segment(1, "source segment 1 ."),
            segment(2, "source segment 2-1 ."),
            segment(2, "source segment 2-2 ."),
            segment(3, "source segment 3 ."),
        ],
    )
    target_text = MemoryText(
        "text1",
        [
            segment(1, "target segment 1 ."),
            segment(2, "target segment 2 ."),
        ],
    )

    parallel_text = ParallelText(source_text, target_text, MemoryTextAlignmentCollection("text1"))
    segments = list(parallel_text.get_segments(all_source_segments=True))
    assert len(segments) == 4
    assert segments[1].source_segment_ref == TextSegmentRef(2)
    assert segments[1].target_segment_ref == TextSegmentRef(2)
    assert segments[1].source_segment == "source segment 2-1 .".split()
    assert segments[1].target_segment == "target segment 2 .".split()
    assert segments[2].source_segment_ref == TextSegmentRef(2)
    assert segments[2].target_segment_ref == TextSegmentRef(2)
    assert segments[2].source_segment == "source segment 2-2 .".split()
    assert segments[2].target_segment == "target segment 2 .".split()


def segment(
    key: int, text: str = "", is_sentence_start: bool = True, is_in_range: bool = False, is_range_start: bool = False
) -> TextSegment:
    return TextSegment(
        "text1",
        TextSegmentRef(key),
        [] if len(text) == 0 else text.split(),
        is_sentence_start=is_sentence_start,
        is_in_range=is_in_range,
        is_range_start=is_range_start,
        is_empty=len(text) == 0,
    )


def alignment(key: int, *pairs: AlignedWordPair) -> TextAlignment:
    return TextAlignment("text1", TextSegmentRef(key), set(pairs))
