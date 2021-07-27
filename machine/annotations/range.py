from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, Optional, Sized, TypeVar, cast

Offset = TypeVar("Offset")


@dataclass(frozen=True)
class Range(Generic[Offset], Sized, Iterable[Offset]):
    _factory: "_RangeFactory[Offset]"
    start: Offset
    end: Offset

    @property
    def length(self) -> int:
        return self._factory.get_length(self.start, self.end)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Offset]:
        return iter(self._factory.iterate(self.start, self.end))


class _RangeFactory(ABC, Generic[Offset]):
    def create(self, start: Offset, end: Optional[Offset]) -> Range[Offset]:
        if end is None:
            end = start
        return Range(self, start, end)

    @abstractmethod
    def get_length(self, start: Offset, end: Offset) -> int:
        ...

    @abstractmethod
    def iterate(self, start: Offset, end: Offset) -> Iterable[Offset]:
        ...


class _IntRangeFactory(_RangeFactory[int]):
    def create(self, start: int, end: Optional[int]) -> "Range[int]":
        if end is None:
            end = start + 1
        return Range(self, start, end)

    def get_length(self, start: int, end: int) -> int:
        return end - start

    def iterate(self, start: int, end: int) -> Iterable[int]:
        return range(start, end)


_INT_RANGE_FACTORY = _IntRangeFactory()


def create_range(start: Offset, end: Optional[Offset] = None) -> Range[Offset]:
    if isinstance(start, int):
        factory = cast(_RangeFactory[Offset], _INT_RANGE_FACTORY)
    else:
        raise RuntimeError("Range type not supported.")

    return factory.create(start, end)
