from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable, List, Sequence, Union, overload

from ..utils.comparable import Comparable, compare
from ..utils.string_utils import parse_integer


@dataclass(frozen=True)
class TextCorpusRowRef(Comparable):
    keys: Sequence[str]

    @overload
    def __init__(self, keys: Iterable[Union[str, int]]) -> None:
        ...

    @overload
    def __init__(self, *keys: Union[str, int]) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        keys: List[str]
        if len(args) == 0:
            keys = []
        elif isinstance(args[0], (str, int)):
            keys = [str(i) for i in args]
        else:
            keys = [str(i) for i in args[0]]
        object.__setattr__(self, "keys", keys)

    def compare_to(self, other: object) -> int:
        if not isinstance(other, TextCorpusRowRef):
            raise TypeError("other is not a TextCorpusRowRef object.")
        if self is other:
            return 0
        for i in range(min(len(self.keys), len(other.keys))):
            key = self.keys[i]
            other_key = other.keys[i]
            if key != other_key:
                # if both keys are numbers, compare numerically
                int_key = parse_integer(key)
                int_other_key = parse_integer(other_key)
                if int_key is not None and int_other_key is not None:
                    return compare(int_key, int_other_key)
                return compare(key, other_key)
        return compare(len(self.keys), len(other.keys))

    def __hash__(self) -> int:
        def reduce_func(code: int, item: str) -> int:
            return code * 31 + hash(item)

        return reduce(reduce_func, self.keys, 23)

    def __repr__(self) -> str:
        return ".".join(self.keys)
