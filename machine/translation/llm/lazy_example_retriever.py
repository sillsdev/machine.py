from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union

from .example import Example


class LazyExampleRetriever(ABC):

    def __init__(self, examples: Iterable[Example]) -> None:
        self._examples = examples
        self._fitted = False

    @property
    def examples(self) -> List[Example]:
        if not isinstance(self._examples, List):
            self._examples = list(self._examples)
        return self._examples

    @abstractmethod
    def _fit(self) -> None: ...

    def rank(self, query: str, k: int, exclude: Optional[Union[int, Example]] = None) -> List[Example]:
        if k <= 0:
            return []
        num_examples = len(self.examples)
        exclude_index = exclude if isinstance(exclude, int) or exclude is None else self.examples.index(exclude)
        if k >= num_examples:
            return [example for index, example in enumerate(self.examples) if index != exclude_index]
        if not self._fitted:
            self._fit()
        self._fitted = True
        return self._rank(query, k, exclude_index)

    @abstractmethod
    def _rank(self, query: str, k: int, exclude_index: Optional[int] = None) -> List[Example]: ...
