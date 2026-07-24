from machine.corpora import DictionaryTextCorpus, MemoryText, TextRow, flatten


def test_flatten_text_corpus() -> None:
    corpus = flatten(
        [_create_text_corpus("text1", ["source 1", "source 2"]), _create_text_corpus("text2", ["source 3"])]
    )

    rows = list(corpus.get_rows())
    assert len(rows) == 3
    assert [row.text for row in rows] == ["source 1", "source 2", "source 3"]
    # the corpus is re-iterable
    assert len(list(corpus.get_rows())) == 3


def test_flatten_parallel_text_corpus() -> None:
    corpus1 = _create_text_corpus("text1", ["source 1"]).align_rows(_create_text_corpus("text1", ["target 1"]))
    corpus2 = _create_text_corpus("text2", ["source 2"]).align_rows(_create_text_corpus("text2", ["target 2"]))
    corpus = flatten([corpus1, corpus2])

    rows = list(corpus.get_rows())
    assert len(rows) == 2
    assert [row.source_text for row in rows] == ["source 1", "source 2"]
    assert [row.target_text for row in rows] == ["target 1", "target 2"]
    # the corpus is re-iterable
    assert len(list(corpus.get_rows())) == 2


def test_flatten_parallel_text_corpus_different_classes() -> None:
    corpus1 = _create_text_corpus("text1", ["Source 1"]).align_rows(_create_text_corpus("text1", ["Target 1"]))
    corpus2 = _create_text_corpus("text2", ["Source 2"]).align_rows(_create_text_corpus("text2", ["Target 2"]))
    corpus = flatten([corpus1, corpus2.lowercase()])

    rows = list(corpus.get_rows())
    assert [row.source_text for row in rows] == ["Source 1", "source 2"]


def test_flatten_parallel_text_corpus_text_ids() -> None:
    corpus1 = _create_text_corpus("text1", ["source 1"]).align_rows(_create_text_corpus("text1", ["target 1"]))
    corpus2 = _create_text_corpus("text2", ["source 2"]).align_rows(_create_text_corpus("text2", ["target 2"]))
    corpus = flatten([corpus1, corpus2])

    rows = list(corpus.get_rows(["text2"]))
    assert [row.source_text for row in rows] == ["source 2"]


def _create_text_corpus(text_id: str, sentences: list) -> DictionaryTextCorpus:
    return DictionaryTextCorpus(
        MemoryText(text_id, [TextRow(text_id, i, [sentence]) for i, sentence in enumerate(sentences)])
    )
