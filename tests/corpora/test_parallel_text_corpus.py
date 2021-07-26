from machine.corpora import (
    DictionaryTextAlignmentCorpus,
    DictionaryTextCorpus,
    MemoryText,
    MemoryTextAlignmentCollection,
    ParallelTextCorpus,
)


def test_texts_no_texts() -> None:
    source_corpus = DictionaryTextCorpus()
    target_corpus = DictionaryTextCorpus()

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus)
    assert not any(parallel_corpus.texts)


def test_texts_no_missing_texts() -> None:
    source_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"), MemoryText("text3"))
    target_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"), MemoryText("text3"))
    alignment_corpus = DictionaryTextAlignmentCorpus(
        MemoryTextAlignmentCollection("text1"),
        MemoryTextAlignmentCollection("text2"),
        MemoryTextAlignmentCollection("text3"),
    )

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    texts = parallel_corpus.texts
    assert [t.id for t in texts] == ["text1", "text2", "text3"]


def test_texts_missing_text() -> None:
    source_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"), MemoryText("text3"))
    target_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text3"))
    alignment_corpus = DictionaryTextAlignmentCorpus(
        MemoryTextAlignmentCollection("text1"), MemoryTextAlignmentCollection("text3")
    )

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    texts = parallel_corpus.texts
    assert [t.id for t in texts] == ["text1", "text3"]


def test_get_texts_missing_target_text_all_source_segments() -> None:
    source_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"), MemoryText("text3"))
    target_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text3"))
    alignment_corpus = DictionaryTextAlignmentCorpus(
        MemoryTextAlignmentCollection("text1"), MemoryTextAlignmentCollection("text3")
    )

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    texts = parallel_corpus.get_texts(all_source_segments=True)
    assert [t.id for t in texts] == ["text1", "text2", "text3"]


def test_get_texts_missing_source_text_all_target_segments() -> None:
    source_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text3"))
    target_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"), MemoryText("text3"))
    alignment_corpus = DictionaryTextAlignmentCorpus(
        MemoryTextAlignmentCollection("text1"), MemoryTextAlignmentCollection("text3")
    )

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    texts = parallel_corpus.get_texts(all_target_segments=True)
    assert [t.id for t in texts] == ["text1", "text2", "text3"]


def test_get_texts_missing_source_and_target_text_all_source_and_target_segments() -> None:
    source_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text3"))
    target_corpus = DictionaryTextCorpus(MemoryText("text1"), MemoryText("text2"))
    alignment_corpus = DictionaryTextAlignmentCorpus(MemoryTextAlignmentCollection("text1"))

    parallel_corpus = ParallelTextCorpus(source_corpus, target_corpus, alignment_corpus)
    texts = parallel_corpus.get_texts(all_source_segments=True, all_target_segments=True)
    assert [t.id for t in texts] == ["text1", "text2", "text3"]
