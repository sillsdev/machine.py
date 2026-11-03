from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from machine.translation.llm.example import Example

from .example_retriever import ExampleRetriever


class TopKExampleRetriever(ExampleRetriever, ABC):
    @abstractmethod
    def _score_examples(self, query: str) -> np.ndarray: ...

    def _rank(self, query: str, k: int, exclude_indices: Optional[List[int]] = None) -> List[Example]:
        scores = self._score_examples(query)
        if len(scores) == 0:
            return [
                example
                for i, example in enumerate(self._examples)
                if exclude_indices is None or i not in exclude_indices
            ][:k]
        n = scores.shape[0]
        available = n - len(exclude_indices) if exclude_indices is not None else 0
        k = min(k, available)
        if k <= 0:
            return []
        if exclude_indices is not None:
            scores = scores.copy()
            for index in exclude_indices:
                scores[index] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        top_indices: List[int] = top[np.argsort(-scores[top])].tolist()
        return [self._examples[i] for i in top_indices]
