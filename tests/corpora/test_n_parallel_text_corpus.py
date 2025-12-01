from machine.corpora import DictionaryTextCorpus, MemoryText, NParallelTextCorpus, TextRow, TextRowFlags


def test_get_rows_zero_corpora() -> None:
    n_parallel_corpus = NParallelTextCorpus([])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 0


def test_get_rows_three_corpora():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert all([r[0] == 1 for r in rows[0].n_refs])
    assert all([r == "source segment 1 .".split() for r in rows[0].n_segments])
    assert not rows[0].is_sentence_start(0)
    assert rows[0].is_sentence_start(1)
    assert all([r[0] == 3 for r in rows[2].n_refs])
    assert all([r == "source segment 3 .".split() for r in rows[2].n_segments])
    assert not rows[2].is_sentence_start(1)
    assert rows[2].is_sentence_start(2)


def test_get_rows_three_corpora_missing_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 1
    assert all([r[0] == 3 for r in rows[0].n_refs])
    assert all([r == "source segment 3 .".split() for r in rows[0].n_segments])
    assert rows[0].is_sentence_start(0)
    assert not rows[0].is_sentence_start(1)


def test_get_rows_three_corpora_missing_rows_all_all_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, True, True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert all([r[0] == 3 for r in rows[2].n_refs])
    assert all([r == "source segment 3 .".split() for r in rows[2].n_segments])
    assert rows[2].is_sentence_start(0)
    assert not rows[2].is_sentence_start(1)


def test_get_rows_three_corpora_missing_rows_some_all_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, False, True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 2
    assert all([r[0] == 3 for r in rows[1].n_refs])
    assert all([r == "source segment 3 .".split() for r in rows[1].n_segments])
    assert rows[1].is_sentence_start(0)
    assert not rows[1].is_sentence_start(1)


def test_get_rows_three_corpora_missing_rows_all_all_rows_missing_middle():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 .", TextRowFlags.NONE),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, True, True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert all([len(r) == 0 or r[0] == 2 for r in rows[1].n_refs])
    assert all([len(r) == 0 or r == "source segment 2 .".split() for r in rows[1].n_segments])
    assert rows[1].is_sentence_start(1)


def test_get_rows_three_corpora_missing_rows_missing_last_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, False, False]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert all([r[0] == 2 for r in rows[1].n_refs])
    assert all([len(r) == 0 or r == "source segment 2 .".split() for r in rows[1].n_segments])
    assert rows[1].is_sentence_start(0)


def test_get_rows_three_corpora_one_corpus():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 .", TextRowFlags.NONE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1])
    n_parallel_corpus.all_rows = [True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 2
    assert all([r[0] == 1 for r in rows[0].n_refs])
    assert all([r == "source segment 1 .".split() for r in rows[0].n_segments])
    assert not rows[0].is_sentence_start(0)


def test_get_rows_three_corpora_range():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 3, flags=TextRowFlags.IN_RANGE),
                _text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
                _text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
                _text_row("text1", 4, "source segment 4 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert all([r == [2, 3] for r in rows[1].n_refs])
    assert rows[1].n_segments[0] == "source segment 2 . source segment 3 .".split()


def test_get_rows_three_corpora_overlapping_ranges():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 3, flags=TextRowFlags.IN_RANGE),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row(
                    "text1",
                    1,
                    "source segment 1 . source segment 2 .",
                    TextRowFlags.SENTENCE_START | TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 2, flags=TextRowFlags.IN_RANGE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 1


def test_get_rows_three_corpora_overlapping_ranges_all_individual_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 3, flags=TextRowFlags.IN_RANGE),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row(
                    "text1",
                    1,
                    "source segment 1 . source segment 2 .",
                    TextRowFlags.SENTENCE_START | TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 2, flags=TextRowFlags.IN_RANGE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [False, False, True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 3
    assert rows[0].n_refs[0] == [1]


def test_get_rows_three_corpora_overlapping_ranges_all_one_through_two_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 3, flags=TextRowFlags.IN_RANGE),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row(
                    "text1",
                    1,
                    "source segment 1 . source segment 2 .",
                    TextRowFlags.SENTENCE_START | TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 2, flags=TextRowFlags.IN_RANGE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [False, True, False]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 2
    assert rows[0].n_refs[0] == [1, 2]


def test_get_rows_three_corpora_overlapping_ranges_all_two_through_three_rows():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row(
                    "text1",
                    2,
                    "source segment 2 . source segment 3 .",
                    TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 3, flags=TextRowFlags.IN_RANGE),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row(
                    "text1",
                    1,
                    "source segment 1 . source segment 2 .",
                    TextRowFlags.SENTENCE_START | TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START,
                ),
                _text_row("text1", 2, flags=TextRowFlags.IN_RANGE),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, False, False]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 2
    assert rows[0].n_refs[0] == [1]


def test_get_rows_three_corpora_same_ref_many_to_many():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2-1 ."),
                _text_row("text1", 2, "source segment 2-2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2-1 ."),
                _text_row("text1", 2, "source segment 2-2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2-1 ."),
                _text_row("text1", 2, "source segment 2-2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 10


def test_get_rows_three_corpora_same_ref_corpora_of_different_sizes():
    corpus1 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 2, "source segment 2 ."),
            ],
        )
    )
    corpus2 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
                _text_row("text1", 2, "source segment 2-1 ."),
                _text_row("text1", 2, "source segment 2-2 ."),
                _text_row("text1", 3, "source segment 3 ."),
            ],
        )
    )
    corpus3 = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _text_row("text1", 1, "source segment 1 ."),
            ],
        )
    )
    n_parallel_corpus = NParallelTextCorpus([corpus1, corpus2, corpus3])
    n_parallel_corpus.all_rows = [True, True, True]
    rows = list(n_parallel_corpus.get_rows())
    assert len(rows) == 4
    assert rows[0].n_refs[1] == [1]


def _text_row(text_id: str, row_ref: object, text="", flags=TextRowFlags.SENTENCE_START) -> TextRow:

    tr = TextRow(text_id, row_ref)
    tr.segment = [] if len(text) == 0 else text.split()
    tr.flags = flags
    return tr
