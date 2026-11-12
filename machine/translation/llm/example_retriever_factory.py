from typing import List

from .coverage_example_retriever import CoverageExampleRetriever
from .example import Example
from .example_retriever import ExampleRetriever
from .example_retriever_method import ExampleRetrievalMethod
from .tfidf_example_retriever import TfidfExampleRetriever


class ExampleRetrieverFactory:
    @staticmethod
    def create_example_retriever(method: ExampleRetrievalMethod, examples: List[Example]) -> ExampleRetriever:
        match method:
            case ExampleRetrievalMethod.TFIDF:
                return TfidfExampleRetriever(examples)
            case ExampleRetrievalMethod.COVERAGE:
                return CoverageExampleRetriever(examples)
            case _:
                raise ValueError(f"Unable to create example retriever with method {method}")
