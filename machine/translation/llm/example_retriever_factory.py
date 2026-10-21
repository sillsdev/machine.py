from typing import Iterable, List

from .example import Example
from .example_retriever_method import ExampleRetrievalMethod
from .lazy_example_retriever import LazyExampleRetriever
from .tfidf_example_retriever import TfidfExampleRetriever


class ExampleRetrieverFactory:
    @staticmethod
    def create_example_retriever(method: ExampleRetrievalMethod, examples: Iterable[Example]) -> LazyExampleRetriever:
        match method:
            case ExampleRetrievalMethod.TFIDF:
                return TfidfExampleRetriever(examples)
