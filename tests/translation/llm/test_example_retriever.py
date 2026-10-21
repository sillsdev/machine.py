from testutils.mock_example_retriever import MockExampleRetriever

from machine.translation.llm.example import Example
from machine.translation.llm.example_pool import ExamplePool


def test_example_pool_k_less_than_n():
    examples = [
        Example("This is the source.", "This is the target."),
        Example("Another kind of source.", "And another kind of target."),
    ]
    pool = ExamplePool(retriever=MockExampleRetriever(examples))
    retrieved_examples = pool.select("This is a query.", 1)
    assert retrieved_examples == [examples[1]]
    retrieved_examples = pool.select("Another example.", 1)
    assert retrieved_examples == [examples[0]]


def test_example_pool_excludes_its_entry_when_k_covers_the_pool():
    examples = [Example("Example one", "1"), Example("Example two", "2"), Example("Example three", "3")]
    pool = ExamplePool(retriever=MockExampleRetriever(examples))

    assert pool.select("Query", 99, exclude_index=1) == [examples[0], examples[2]]
