from abc import ABC, abstractmethod
from typing import TypeVar


class Comparable(ABC):
    @abstractmethod
    def compare_to(self, other: object) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        try:
            return self.compare_to(other) == 0
        except TypeError:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        try:
            return self.compare_to(other) < 0
        except TypeError:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        try:
            return self.compare_to(other) > 0
        except TypeError:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        try:
            return self.compare_to(other) <= 0
        except TypeError:
            return NotImplemented

    def __ge__(self, other: object) -> bool:
        try:
            return self.compare_to(other) >= 0
        except TypeError:
            return NotImplemented


T = TypeVar("T", str, int, float, Comparable)


def compare(x: T, y: T) -> int:
    if isinstance(x, Comparable):
        return x.compare_to(y)
    if x < y:
        return -1
    if x > y:
        return 1
    return 0
