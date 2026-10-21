from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from machine.translation.llm.example import Example

from .lazy_example_retriever import LazyExampleRetriever


class TopKExampleRetriever(LazyExampleRetriever, ABC):
    @abstractmethod
    def _score_examples(self, query: str) -> np.ndarray: ...

    def _rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]:
        scores = self._score_examples(query)
        n = scores.shape[0]
        available = n - 1 if exclude_index is not None else n
        k = min(k, available)
        if k <= 0:
            return []
        if exclude_index is not None:
            scores = scores.copy()
            scores[exclude_index] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        top_indices: List[int] = top[np.argsort(-scores[top])].tolist()
        return [self.examples[i] for i in top_indices]
