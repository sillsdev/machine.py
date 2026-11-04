import pytest

from machine.translation.llm import (
    CoverageExampleRetriever,
    Example,
    ExampleRetrievalMethod,
    ExampleRetrieverFactory,
    TfidfExampleRetriever,
)


def test_example_retriever_factory_creates_tfidf_retriever_for_tfidf_method():
    examples = [Example("a", "0"), Example("b", "1")]

    retriever = ExampleRetrieverFactory.create_example_retriever(ExampleRetrievalMethod.TFIDF, examples)

    assert isinstance(retriever, TfidfExampleRetriever)
    assert retriever._examples == examples


def test_example_retriever_factory_creates_coverage_retriever_for_coverage_method():
    examples = [Example("a", "0"), Example("b", "1")]

    retriever = ExampleRetrieverFactory.create_example_retriever(ExampleRetrievalMethod.COVERAGE, examples)

    assert isinstance(retriever, CoverageExampleRetriever)
    assert retriever._examples == examples


def test_example_retriever_factory_raises_for_unknown_method():
    with pytest.raises(ValueError, match="Unable to create example retriever with method"):
        ExampleRetrieverFactory.create_example_retriever("not-a-method", [])  # type: ignore
