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
    retrieved_examples = retriever.rank("query", 2, exclude_index=10)
    assert retrieved_examples == examples


def test_example_retriever_excludes_only_the_specified_index():
    duplicate = Example("a", "1")
    examples = [duplicate, Example("b", "2"), duplicate]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 3, exclude_index=0)
    assert retrieved_examples == examples[1:]


def test_example_retriever_examples_returns_a_copy():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)

    exposed_examples = retriever.examples
    exposed_examples.clear()

    assert retriever.examples == examples
