from typing import Iterable, List, Optional, Union

from .example import Example
from .example_retriever_factory import ExampleRetrieverFactory
from .example_retriever_method import ExampleRetrievalMethod
from .lazy_example_retriever import LazyExampleRetriever


class ExamplePool:
    def __init__(
        self,
        examples: Optional[Iterable[Example]] = None,
        retrieval_method: Optional[ExampleRetrievalMethod] = None,
        retriever: Optional[LazyExampleRetriever] = None,
    ) -> None:
        if (examples is None or retrieval_method is None) and retriever is None:
            raise ValueError("Either examples and retrieval method or a retreiver must be specified.")
        retrieval_method = retrieval_method or ExampleRetrievalMethod.TFIDF
        examples = examples or []
        self._retriever = (
            retriever
            if retriever is not None
            else ExampleRetrieverFactory.create_example_retriever(retrieval_method, examples)
        )

    def select(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]:
        return self._retriever.rank(query, k, exclude_index)
