from abc import ABC, abstractmethod
from typing import List, Optional

from .example import Example


class ExampleRetriever(ABC):

    def __init__(self, examples: List[Example]) -> None:
        self._examples = list(examples)
        self._fit()

    @abstractmethod
    def _fit(self) -> None: ...

    @property
    def examples(self) -> List[Example]:
        return list(self._examples)

    def rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]:
        if k <= 0:
            return []
        num_examples = len(self._examples)
        if exclude_index is not None and not 0 <= exclude_index < num_examples:
            exclude_index = None
        return self._rank(query, k, exclude_index)

    @abstractmethod
    def _rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]: ...
