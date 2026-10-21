from machine.translation.llm import Example
from machine.translation.llm.tfidf_example_retriever import TfidfExampleRetriever


def test_tfidf_retriever_ranks_most_similar_source_first():
    examples = [
        Example("the cat sat on the mat", "0"),
        Example("completely unrelated sentence", "1"),
        Example("a cat sat here", "2"),
    ]

    assert TfidfExampleRetriever(examples).rank("the cat sat", 2) == [examples[0], examples[2]]


def test_tfidf_retriever_respects_k():
    examples = [Example("apple pie", str(index)) for index in range(3)]

    assert len(TfidfExampleRetriever(examples).rank("apple pie", 2)) == 2


def test_tfidf_retriever_excludes_its_own_position():
    examples = [Example("the cat sat", "0"), Example("the cat sat", "1"), Example("totally different words", "2")]

    assert TfidfExampleRetriever(examples).rank("the cat sat", 2, exclude=0) == [examples[1], examples[2]]


def test_tfidf_retriever_k_larger_than_the_corpus_returns_every_example():
    examples = [Example("apple pie", "0"), Example("banana split", "1")]

    assert TfidfExampleRetriever(examples).rank("apple pie", 10) == examples


def test_tfidf_retriever_empty_corpus_returns_no_examples():
    assert TfidfExampleRetriever([]).rank("anything", 5) == []


def test_tfidf_retriever_handles_a_corpus_with_no_word_tokens():
    examples = [Example("!!!", "0"), Example("...", "1")]

    assert TfidfExampleRetriever(examples).rank("anything", 1) == []


def test_tfidf_retriever_indexes_a_corpus_where_only_some_sources_have_words():
    examples = [Example("!!!", "0"), Example("let there be light", "1")]

    assert TfidfExampleRetriever(examples).rank("light", 1) == [examples[1]]


def test_punctuation_carries_no_tfidf_retrieval_signal():
    examples = [
        Example("the cat sat", "0"),
        Example("!!! !!! !!! ???", "1"),
        Example("something else entirely", "2"),
    ]

    assert TfidfExampleRetriever(examples).rank("!!! !!! !!! cat", 1) == [examples[0]]
