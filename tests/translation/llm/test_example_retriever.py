from testutils.mock_example_retriever import MockExampleRetriever

from machine.translation.llm.example import Example


def test_example_retriever_k_greater_than_n():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 3)
    assert retrieved_examples == examples


def test_example_retriever_no_such_exclude():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 2, Example("c", "3"))
    assert retrieved_examples == examples


def test_example_retriever_exclude_example():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 2, Example("a", "1"))
    assert retrieved_examples == examples[1:]
