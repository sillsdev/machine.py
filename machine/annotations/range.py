from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, Optional, Sized, TypeVar, cast

Offset = TypeVar("Offset")


@dataclass(frozen=True)
class Range(Generic[Offset], Sized, Iterable[Offset]):
    _factory: "_RangeFactory[Offset]"
    start: Offset
    end: Offset

    @classmethod
    def create(cls, start: Offset, end: Optional[Offset] = None) -> "Range[Offset]":
        if isinstance(start, int):
            factory = cast(_RangeFactory[Offset], _INT_RANGE_FACTORY)
        else:
            raise RuntimeError("Range type not supported.")

        return factory.create(start, end)

    @property
    def length(self) -> int:
        return self._factory.get_length(self.start, self.end)

    def overlaps(self, other: "Range[Offset]") -> bool:
        if self._factory.include_endpoint:
            return (
                self._factory.offset_compare(self.start, other.end) <= 0
                and self._factory.offset_compare(self.end, other.start) >= 0
            )
        return (
            self._factory.offset_compare(self.start, other.end) < 0
            and self._factory.offset_compare(self.end, other.start) > 0
        )

    def contains(self, other: "Range[Offset]") -> bool:
        return (
            self._factory.offset_compare(self.start, other.start) <= 0
            and self._factory.offset_compare(self.end, other.end) >= 0
        )

    def compare_to(self, other: "Range[Offset]") -> int:
        res = self._factory.offset_compare(self.start, other.start)
        if res == 0:
            res = -self._factory.offset_compare(self.end, other.end)
        return res

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Range) or self._factory != other._factory:
            raise NotImplementedError
        return self.compare_to(other) < 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Range) or self._factory != other._factory:
            raise NotImplementedError
        return self.compare_to(other) > 0

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Range) or self._factory != other._factory:
            raise NotImplementedError
        return self.compare_to(other) <= 0

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Range) or self._factory != other._factory:
            raise NotImplementedError
        return self.compare_to(other) >= 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Offset]:
        return iter(self._factory.iterate(self.start, self.end))

    def __repr__(self) -> str:
        return f"[{self.start}, {self.end}]"


class _RangeFactory(ABC, Generic[Offset]):
    @property
    @abstractmethod
    def include_endpoint(self) -> bool:
        ...

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

    @abstractmethod
    def offset_compare(self, x: Offset, y: Offset) -> int:
        ...


class _IntRangeFactory(_RangeFactory[int]):
    @property
    def include_endpoint(self) -> bool:
        return False

    def create(self, start: int, end: Optional[int]) -> "Range[int]":
        if end is None:
            end = start + 1
        return Range(self, start, end)

    def get_length(self, start: int, end: int) -> int:
        return end - start

    def iterate(self, start: int, end: int) -> Iterable[int]:
        return range(start, end)

    def offset_compare(self, x: int, y: int) -> int:
        if x < y:
            return -1
        if x > y:
            return 1
        return 0


_INT_RANGE_FACTORY = _IntRangeFactory()
