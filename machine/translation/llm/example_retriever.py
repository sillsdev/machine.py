from abc import ABC, abstractmethod
from typing import List, Optional

from .example import Example


class ExampleRetriever(ABC):

    def __init__(self, examples: List[Example]) -> None:
        self._examples = list(examples)
        self._fit()

    @abstractmethod
    def _fit(self) -> None: ...

    def rank(self, query: str, k: int, exclude_example: Optional[Example] = None) -> List[Example]:
        if k <= 0:
            return []
        num_examples = len(self._examples)
        exclude_indices = (
            [i for i, example in enumerate(self._examples) if example == exclude_example]
            if exclude_example is not None
            else []
        )
        if k >= (num_examples - len(exclude_indices)):
            return [example for example in self._examples if example != exclude_example]
        return self._rank(query, k, exclude_indices)

    @abstractmethod
    def _rank(self, query: str, k: int, exclude_indices: Optional[List[int]] = None) -> List[Example]: ...
