from __future__ import annotations

from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


class AlignmentCell(Sequence[T]):
    def __init__(self, items: Iterable[T] = []) -> None:
        item_list = list(items)
        self._items = None
        if len(item_list) > 0:
            self._items = item_list

    @property
    def is_null(self) -> bool:
        return self._items is None

    @property
    def first(self) -> T:
        if self._items is None or len(self._items) == 0:
            raise RuntimeError("The alignment cell is empty.")
        return self._items[0]

    @property
    def last(self) -> T:
        if self._items is None or len(self._items) == 0:
            raise RuntimeError("The alignment cell is empty.")
        return self._items[-1]

    def __len__(self) -> int:
        if self._items is None:
            return 0
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        if self._items is None:
            raise IndexError()
        return self._items[index]

    def __contains__(self, value: T) -> bool:
        if self._items is None:
            return False
        return value in self._items

    def __iter__(self) -> Iterator[T]:
        if self._items is None:
            return iter([])
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        if self._items is None:
            return iter([])
        return reversed(self._items)

    def __eq__(self, other: AlignmentCell[T]) -> bool:
        return self._items == other._items
