from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable, List, Sequence, overload


@dataclass(frozen=True)
class TextSegmentRef:
    keys: Sequence[str]

    @overload
    def __init__(self, keys: Iterable[str]) -> None:
        ...

    @overload
    def __init__(self, key: Iterable[int]) -> None:
        ...

    @overload
    def __init__(self, *args: str) -> None:
        ...

    @overload
    def __init__(self, *args: int) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        keys: List[str]
        if len(args) == 0:
            keys = []
        elif isinstance(args[0], str):
            keys = list(args)
        elif isinstance(args[0], int):
            keys = [str(i) for i in args]
        elif isinstance(args[0][0], str):
            keys = list(args[0])
        else:
            keys = [str(i) for i in args[0]]
        object.__setattr__(self, "keys", keys)

    def __hash__(self) -> int:
        def reduce_func(code: int, item: str) -> int:
            return code * 31 + hash(item)

        return reduce(reduce_func, self.keys, 23)

    def __lt__(self, other: "TextSegmentRef") -> bool:
        for i in range(min(len(self.keys), len(other.keys))):
            key = self.keys[i]
            other_key = other.keys[i]
            if key != other_key:
                # if both keys are numbers, compare numerically
                if key.isdigit() and other_key.isdigit():
                    return int(key) < int(other_key)
                return key < other_key
        return len(self.keys) < len(other.keys)

    def __le__(self, other: "TextSegmentRef") -> bool:
        return not (self > other)

    def __gt__(self, other: "TextSegmentRef") -> bool:
        for i in range(min(len(self.keys), len(other.keys))):
            key = self.keys[i]
            other_key = other.keys[i]
            if key != other_key:
                # if both keys are numbers, compare numerically
                if key.isdigit() and other_key.isdigit():
                    return int(key) > int(other_key)
                return key > other_key
        return len(self.keys) > len(other.keys)

    def __ge__(self, other: "TextSegmentRef") -> bool:
        return not (self < other)

    def __repr__(self) -> str:
        return ".".join(self.keys)
