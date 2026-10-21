import numpy as np

from machine.translation.llm.top_k_example_retriever import TopKExampleRetriever


class MockExampleRetriever(TopKExampleRetriever):
    def _fit(self) -> None:
        pass

    def _score_examples(self, query: str) -> np.ndarray:
        return np.array([0 if query == e.source else (1 if query > e.source else -1) for e in self.examples])
