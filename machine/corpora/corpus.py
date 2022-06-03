from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Callable, Generator, Generic, Iterable, Optional, Sequence, Tuple, TypeVar

from ..utils.context_managed_generator import ContextManagedGenerator
from .alignment_row import AlignmentRow
from .corpora_utils import batch, get_split_indices
from .parallel_text_row import ParallelTextRow
from .text_row import TextRow

Row = TypeVar("Row", TextRow, ParallelTextRow, AlignmentRow)
Item = TypeVar("Item")


class Corpus(ABC, Generic[Row], Iterable[Row]):
    def get_rows(self) -> ContextManagedGenerator[Row, None, None]:
        return ContextManagedGenerator(self._get_rows())

    @abstractmethod
    def _get_rows(self) -> Generator[Row, None, None]:
        ...

    def __iter__(self) -> ContextManagedGenerator[Row, None, None]:
        return self.get_rows()

    @property
    def missing_rows_allowed(self) -> bool:
        return True

    def count(self, include_empty: bool = True) -> int:
        with self.get_rows() as rows:
            return sum(1 for row in rows if include_empty or not row.is_empty)

    def interleaved_split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple[ContextManagedGenerator[Tuple[Row, bool], None, None], int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        def _get_rows() -> Generator[Tuple[Row, bool], None, None]:
            with self.get_rows() as rows:
                yield from (
                    (row, i in split_indices) for i, row in enumerate(rows) if include_empty or not row.is_empty
                )

        return (
            ContextManagedGenerator(_get_rows()),
            corpus_size - len(split_indices),
            len(split_indices),
        )

    def take(self, count: int) -> ContextManagedGenerator[Row, None, None]:
        def _get_rows() -> Generator[Row, None, None]:
            with self.get_rows() as rows:
                yield from islice(rows, count)

        return ContextManagedGenerator(_get_rows())

    def map(self, selector: Callable[[Row], Item]) -> ContextManagedGenerator[Item, None, None]:
        return self.map_by_index(lambda r, _: selector(r))

    def map_by_index(self, selector: Callable[[Row, int], Item]) -> ContextManagedGenerator[Item, None, None]:
        def _map_rows() -> Generator[Item, None, None]:
            with self.get_rows() as rows:
                yield from (selector(row, i) for i, row in enumerate(rows))

        return ContextManagedGenerator(_map_rows())

    def batch(self, batch_size: int) -> ContextManagedGenerator[Sequence[Row], None, None]:
        def _batch() -> Generator[Sequence[Row], None, None]:
            with self.get_rows() as rows:
                yield from batch(rows, batch_size)

        return ContextManagedGenerator(_batch())
