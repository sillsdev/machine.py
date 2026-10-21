from testutils.mock_example_retriever import MockExampleRetriever

from machine.translation.llm import Example, ExamplePool, ExampleRetrievalMethod


def test_example_pool_pass_retriever():
    examples = [
        Example("This is the source.", "This is the target."),
        Example("Another kind of source.", "And another kind of target."),
    ]
    pool = ExamplePool(retriever=MockExampleRetriever(examples))
    retrieved_examples = pool.select("This is a query.", 2)
    assert len(retrieved_examples) == 2
    assert retrieved_examples[0].source == "This is the source."
    assert retrieved_examples[1].source == "Another kind of source."


def test_example_pool_pass_method():
    examples = [
        Example("This is the source.", "This is the target."),
        Example("Another kind of source.", "And another kind of target."),
    ]
    pool = ExamplePool(examples, ExampleRetrievalMethod.TFIDF)
    retrieved_examples = pool.select("This is a query.", 1)
    assert len(retrieved_examples) == 1
    assert retrieved_examples[0].source == "This is the source."
