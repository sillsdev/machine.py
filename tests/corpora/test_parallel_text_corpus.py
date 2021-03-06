from io import StringIO
from typing import Any

from machine.corpora import (
    AlignedWordPair,
    AlignmentRow,
    DictionaryAlignmentCorpus,
    DictionaryTextCorpus,
    MemoryAlignmentCollection,
    MemoryText,
    StandardParallelTextCorpus,
    TextRow,
)
from machine.scripture import ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION, VerseRef, Versification


def test_get_rows_no_segments() -> None:
    source_corpus = DictionaryTextCorpus()
    target_corpus = DictionaryTextCorpus()
    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    assert not any(parallel_corpus)


def test_get_rows_no_missing_rows() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 .", is_sentence_start=False),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 .", is_sentence_start=False),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 1, AlignedWordPair(0, 0)),
                alignment_row("text1", 2, AlignedWordPair(1, 1)),
                alignment_row("text1", 3, AlignedWordPair(2, 2)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 3
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()
    assert not rows[0].is_source_sentence_start
    assert rows[0].is_target_sentence_start
    assert rows[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert rows[2].source_refs == [3]
    assert rows[2].target_refs == [3]
    assert rows[2].source_segment == "source segment 3 .".split()
    assert rows[2].target_segment == "target segment 3 .".split()
    assert rows[2].is_source_sentence_start
    assert not rows[2].is_target_sentence_start
    assert rows[2].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_get_rows_missing_middle_target_rows() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 1, AlignedWordPair(0, 0)),
                alignment_row("text1", 3, AlignedWordPair(2, 2)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert rows[1].source_refs == [3]
    assert rows[1].target_refs == [3]
    assert rows[1].source_segment == "source segment 3 .".split()
    assert rows[1].target_segment == "target segment 3 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_get_rows_missing_middle_source_row() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 1, AlignedWordPair(0, 0)),
                alignment_row("text1", 3, AlignedWordPair(2, 2)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert rows[1].source_refs == [3]
    assert rows[1].target_refs == [3]
    assert rows[1].source_segment == "source segment 3 .".split()
    assert rows[1].target_segment == "target segment 3 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_get_rows_missing_last_target_row() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 1, AlignedWordPair(0, 0)),
                alignment_row("text1", 2, AlignedWordPair(1, 1)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2 .".split()
    assert rows[1].target_segment == "target segment 2 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_get_rows_missing_last_source_row() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 1, AlignedWordPair(0, 0)),
                alignment_row("text1", 2, AlignedWordPair(1, 1)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(0, 0)}
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2 .".split()
    assert rows[1].target_segment == "target segment 2 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(1, 1)}


def test_get_rows_missing_first_target_row() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 2, AlignedWordPair(1, 1)),
                alignment_row("text1", 3, AlignedWordPair(2, 2)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [2]
    assert rows[0].target_refs == [2]
    assert rows[0].source_segment == "source segment 2 .".split()
    assert rows[0].target_segment == "target segment 2 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert rows[1].source_refs == [3]
    assert rows[1].target_refs == [3]
    assert rows[1].source_segment == "source segment 3 .".split()
    assert rows[1].target_segment == "target segment 3 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_get_rows_missing_first_source_row() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )
    alignment_corpus = DictionaryAlignmentCorpus(
        MemoryAlignmentCollection(
            "text1",
            [
                alignment_row("text1", 2, AlignedWordPair(1, 1)),
                alignment_row("text1", 3, AlignedWordPair(2, 2)),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [2]
    assert rows[0].target_refs == [2]
    assert rows[0].source_segment == "source segment 2 .".split()
    assert rows[0].target_segment == "target segment 2 .".split()
    assert rows[0].aligned_word_pairs == {AlignedWordPair(1, 1)}
    assert rows[1].source_refs == [3]
    assert rows[1].target_refs == [3]
    assert rows[1].source_segment == "source segment 3 .".split()
    assert rows[1].target_segment == "target segment 3 .".split()
    assert rows[1].aligned_word_pairs == {AlignedWordPair(2, 2)}


def test_get_rows_range() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    is_sentence_start=False,
                    is_in_range=True,
                    is_range_start=True,
                ),
                text_row("text1", 3, is_in_range=True),
                text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
                text_row("text1", 4, "target segment 4 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 3
    assert rows[1].source_refs == [2, 3]
    assert rows[1].target_refs == [2, 3]
    assert rows[1].source_segment == "source segment 2 . source segment 3 .".split()
    assert rows[1].target_segment == "target segment 2 . target segment 3 .".split()
    assert not rows[1].is_source_sentence_start
    assert rows[1].is_target_sentence_start


def test_get_rows_overlapping_ranges() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 . source segment 3 .", is_in_range=True, is_range_start=True),
                text_row("text1", 3, is_in_range=True),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 . target segment 2 .", is_in_range=True, is_range_start=True),
                text_row("text1", 2, is_in_range=True),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 1
    assert rows[0].source_refs == [1, 2, 3]
    assert rows[0].target_refs == [1, 2, 3]
    assert rows[0].source_segment == "source segment 1 . source segment 2 . source segment 3 .".split()
    assert rows[0].target_segment == "target segment 1 . target segment 2 . target segment 3 .".split()
    assert rows[0].is_source_sentence_start
    assert rows[0].is_target_sentence_start


def test_get_rows_adjacent_ranges_same_text() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row(
                    "text1",
                    1,
                    "source segment 1 . source segment 2 .",
                    is_sentence_start=False,
                    is_in_range=True,
                    is_range_start=True,
                ),
                text_row("text1", 2, is_in_range=True),
                text_row("text1", 3, "source segment 3 . source segment 4 .", is_in_range=True, is_range_start=True),
                text_row("text1", 4, is_in_range=True),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 .", is_sentence_start=False),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
                text_row("text1", 4, "target segment 4 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1, 2]
    assert rows[0].target_refs == [1, 2]
    assert rows[0].source_segment == "source segment 1 . source segment 2 .".split()
    assert rows[0].target_segment == "target segment 1 . target segment 2 .".split()
    assert not rows[0].is_source_sentence_start
    assert not rows[0].is_target_sentence_start
    assert rows[1].source_refs == [3, 4]
    assert rows[1].target_refs == [3, 4]
    assert rows[1].source_segment == "source segment 3 . source segment 4 .".split()
    assert rows[1].target_segment == "target segment 3 . target segment 4 .".split()
    assert rows[1].is_source_sentence_start
    assert rows[1].is_target_sentence_start


def test_get_rows_adjacent_ranges_different_texts() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 . source segment 2 .", is_in_range=True, is_range_start=True),
                text_row("text1", 2, is_in_range=True),
                text_row("text1", 3, "source segment 3 ."),
                text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 . target segment 4 .", is_in_range=True, is_range_start=True),
                text_row("text1", 4, is_in_range=True),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1, 2]
    assert rows[0].target_refs == [1, 2]
    assert rows[0].source_segment == "source segment 1 . source segment 2 .".split()
    assert rows[0].target_segment == "target segment 1 . target segment 2 .".split()
    assert rows[1].source_refs == [3, 4]
    assert rows[1].target_refs == [3, 4]
    assert rows[1].source_segment == "source segment 3 . source segment 4 .".split()
    assert rows[1].target_segment == "target segment 3 . target segment 4 .".split()


def test_get_segments_all_source_rows() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
                text_row("text1", 4, "source segment 4 ."),
            ],
        ),
        MemoryText(
            "text2",
            [
                text_row("text2", 5, "source segment 5 ."),
            ],
        ),
        MemoryText(
            "text3",
            [
                text_row("text3", 6, "source segment 6 ."),
                text_row("text3", 7, "source segment 7 ."),
            ],
        ),
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 3, "target segment 3 ."),
                text_row("text1", 4, "target segment 4 ."),
            ],
        ),
        MemoryText(
            "text3",
            [
                text_row("text3", 6, "target segment 6 ."),
                text_row("text3", 7, "target segment 7 ."),
            ],
        ),
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_source_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 7
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == []
    assert rows[1].source_segment == "source segment 2 .".split()
    assert rows[1].target_segment == []

    assert rows[4].source_refs == [5]
    assert rows[4].target_refs == []
    assert rows[4].source_segment == "source segment 5 .".split()
    assert rows[4].target_segment == []


def test_get_segments_missing_text() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [text_row("text1", 1, "source segment 1 .")],
        ),
        MemoryText(
            "text2",
            [text_row("text2", 2, "source segment 2 .")],
        ),
        MemoryText(
            "text3",
            [text_row("text3", 3, "source segment 3 .")],
        ),
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [text_row("text1", 1, "target segment 1 .")],
        ),
        MemoryText(
            "text3",
            [text_row("text3", 3, "target segment 3 .")],
        ),
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 2
    assert rows[0].source_refs == [1]
    assert rows[0].target_refs == [1]
    assert rows[0].source_segment == "source segment 1 .".split()
    assert rows[0].target_segment == "target segment 1 .".split()

    assert rows[1].source_refs == [3]
    assert rows[1].target_refs == [3]
    assert rows[1].source_segment == "source segment 3 .".split()
    assert rows[1].target_segment == "target segment 3 .".split()


def test_get_segments_range_all_target_rows() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 . source segment 3 .", is_in_range=True, is_range_start=True),
                text_row("text1", 3, is_in_range=True),
                text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
                text_row("text1", 4, "target segment 4 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_target_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 4
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2 . source segment 3 .".split()
    assert rows[1].is_source_in_range
    assert rows[1].is_source_range_start
    assert rows[1].target_segment == "target segment 2 .".split()
    assert rows[2].source_refs == [3]
    assert rows[2].target_refs == [3]
    assert not any(rows[2].source_segment)
    assert rows[2].is_source_in_range
    assert not rows[2].is_source_range_start
    assert rows[2].target_segment == "target segment 3 .".split()


def test_get_rows_same_ref_middle_many_to_many() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2-1 ."),
                text_row("text1", 2, "source segment 2-2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2-1 ."),
                text_row("text1", 2, "target segment 2-2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 6
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2-1 .".split()
    assert rows[1].target_segment == "target segment 2-1 .".split()
    assert rows[2].source_refs == [2]
    assert rows[2].target_refs == [2]
    assert rows[2].source_segment == "source segment 2-1 .".split()
    assert rows[2].target_segment == "target segment 2-2 .".split()
    assert rows[3].source_refs == [2]
    assert rows[3].target_refs == [2]
    assert rows[3].source_segment == "source segment 2-2 .".split()
    assert rows[3].target_segment == "target segment 2-1 .".split()
    assert rows[4].source_refs == [2]
    assert rows[4].target_refs == [2]
    assert rows[4].source_segment == "source segment 2-2 .".split()
    assert rows[4].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_one_to_many() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2-1 ."),
                text_row("text1", 2, "target segment 2-2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_target_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 4
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2 .".split()
    assert rows[1].target_segment == "target segment 2-1 .".split()
    assert rows[2].source_refs == [2]
    assert rows[2].target_refs == [2]
    assert rows[2].source_segment == "source segment 2 .".split()
    assert rows[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_middle_many_to_one() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2-1 ."),
                text_row("text1", 2, "source segment 2-2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_source_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 4
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2-1 .".split()
    assert rows[1].target_segment == "target segment 2 .".split()
    assert rows[2].source_refs == [2]
    assert rows[2].target_refs == [2]
    assert rows[2].source_segment == "source segment 2-2 .".split()
    assert rows[2].target_segment == "target segment 2 .".split()


def test_get_segments_same_ref_last_one_to_many() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2-1 ."),
                text_row("text1", 2, "target segment 2-2 ."),
                text_row("text1", 3, "target segment 3 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_target_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 4
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2 .".split()
    assert rows[1].target_segment == "target segment 2-1 .".split()
    assert rows[2].source_refs == [2]
    assert rows[2].target_refs == [2]
    assert rows[2].source_segment == "source segment 2 .".split()
    assert rows[2].target_segment == "target segment 2-2 .".split()


def test_get_segments_same_ref_last_many_to_one() -> None:
    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "source segment 1 ."),
                text_row("text1", 2, "source segment 2-1 ."),
                text_row("text1", 2, "source segment 2-2 ."),
                text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                text_row("text1", 1, "target segment 1 ."),
                text_row("text1", 2, "target segment 2 ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus, all_source_rows=True)
    rows = list(parallel_corpus)
    assert len(rows) == 4
    assert rows[1].source_refs == [2]
    assert rows[1].target_refs == [2]
    assert rows[1].source_segment == "source segment 2-1 .".split()
    assert rows[1].target_segment == "target segment 2 .".split()
    assert rows[2].source_refs == [2]
    assert rows[2].target_refs == [2]
    assert rows[2].source_segment == "source segment 2-2 .".split()
    assert rows[2].target_segment == "target segment 2 .".split()


def test_get_segments_same_verse_ref_one_to_many() -> None:
    stream = StringIO("&MAT 1:2-3 = MAT 1:2\n" "MAT 1:4 = MAT 1:3\n")
    versification = Versification("custom", "vers.txt", ENGLISH_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")

    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "MAT",
            [
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:1", ORIGINAL_VERSIFICATION), "source chapter one, verse one ."
                ),
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:2", ORIGINAL_VERSIFICATION), "source chapter one, verse two ."
                ),
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:3", ORIGINAL_VERSIFICATION), "source chapter one, verse three ."
                ),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "MAT",
            [
                text_row("MAT", VerseRef.from_string("MAT 1:1", versification), "target chapter one, verse one ."),
                text_row(
                    "MAT",
                    VerseRef.from_string("MAT 1:2", versification),
                    "target chapter one, verse two . target chapter one, verse three .",
                    is_in_range=True,
                    is_range_start=True,
                ),
                text_row("MAT", VerseRef.from_string("MAT 1:3", versification), is_in_range=True),
                text_row("MAT", VerseRef.from_string("MAT 1:4", versification), "target chapter one, verse four ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 3
    assert rows[1].source_refs == [VerseRef.from_string("MAT 1:2", ORIGINAL_VERSIFICATION)]
    assert rows[1].target_refs == [
        VerseRef.from_string("MAT 1:2", versification),
        VerseRef.from_string("MAT 1:3", versification),
    ]
    assert rows[1].source_segment == "source chapter one, verse two .".split()
    assert rows[1].target_segment == "target chapter one, verse two . target chapter one, verse three .".split()


def test_get_rows_verse_ref_out_of_order() -> None:
    stream = StringIO("&MAT 1:4-5 = MAT 1:4\nMAT 1:2 = MAT 1:3\nMAT 1:3 = MAT 1:2\n")
    versification = Versification("custom", "vers.txt", ENGLISH_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")

    source_corpus = DictionaryTextCorpus(
        MemoryText(
            "MAT",
            [
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:1", ORIGINAL_VERSIFICATION), "source chapter one, verse one ."
                ),
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:2", ORIGINAL_VERSIFICATION), "source chapter one, verse two ."
                ),
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:3", ORIGINAL_VERSIFICATION), "source chapter one, verse three ."
                ),
                text_row(
                    "MAT", VerseRef.from_string("MAT 1:4", ORIGINAL_VERSIFICATION), "source chapter one, verse four ."
                ),
            ],
        )
    )
    target_corpus = DictionaryTextCorpus(
        MemoryText(
            "MAT",
            [
                text_row("MAT", VerseRef.from_string("MAT 1:1", versification), "target chapter one, verse one ."),
                text_row("MAT", VerseRef.from_string("MAT 1:2", versification), "target chapter one, verse two ."),
                text_row("MAT", VerseRef.from_string("MAT 1:3", versification), "target chapter one, verse three ."),
                text_row("MAT", VerseRef.from_string("MAT 1:4", versification), "target chapter one, verse four ."),
                text_row("MAT", VerseRef.from_string("MAT 1:5", versification), "target chapter one, verse five ."),
            ],
        )
    )

    parallel_corpus = StandardParallelTextCorpus(source_corpus, target_corpus)
    rows = list(parallel_corpus)
    assert len(rows) == 4

    assert rows[1].source_refs == [VerseRef.from_string("MAT 1:2", ORIGINAL_VERSIFICATION)]
    assert rows[1].target_refs == [VerseRef.from_string("MAT 1:3", versification)]
    assert rows[1].source_segment == "source chapter one, verse two .".split()
    assert rows[1].target_segment == "target chapter one, verse three .".split()

    assert rows[2].source_refs == [VerseRef.from_string("MAT 1:3", ORIGINAL_VERSIFICATION)]
    assert rows[2].target_refs == [VerseRef.from_string("MAT 1:2", versification)]
    assert rows[2].source_segment == "source chapter one, verse three .".split()
    assert rows[2].target_segment == "target chapter one, verse two .".split()

    assert rows[3].source_refs == [VerseRef.from_string("MAT 1:4", ORIGINAL_VERSIFICATION)]
    assert rows[3].target_refs == [
        VerseRef.from_string("MAT 1:4", versification),
        VerseRef.from_string("MAT 1:5", versification),
    ]
    assert rows[3].source_segment == "source chapter one, verse four .".split()
    assert rows[3].target_segment == "target chapter one, verse four . target chapter one, verse five .".split()


def text_row(
    text_id: str,
    ref: Any,
    text: str = "",
    is_sentence_start: bool = True,
    is_in_range: bool = False,
    is_range_start: bool = False,
) -> TextRow:
    return TextRow(
        text_id,
        ref,
        [] if len(text) == 0 else text.split(),
        is_sentence_start=is_sentence_start,
        is_in_range=is_in_range,
        is_range_start=is_range_start,
        is_empty=len(text) == 0,
    )


def alignment_row(text_id: str, ref: int, *pairs: AlignedWordPair) -> AlignmentRow:
    return AlignmentRow(text_id, ref, set(pairs))
