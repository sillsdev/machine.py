from dataclasses import dataclass
from typing import Any, List

from ..utils.comparable import Comparable, compare


@dataclass(frozen=True)
class MultiKeyRef(Comparable):
    text_id: str
    keys: List[Any]

    def compare_to(self, other: object) -> int:
        if not isinstance(other, MultiKeyRef):
            raise TypeError("other is not a TextFileRef object.")
        if self is other:
            return 0

        result = compare(self.text_id, other.text_id)
        if result != 0:
            return result

        for key, other_key in zip(self.keys, other.keys):
            result = compare(key, other_key)
            if result != 0:
                return result

        return compare(len(self.keys), len(other.keys))

    def __hash__(self) -> int:
        code = 23
        code = code * 31 + hash(self.text_id)
        for key in self.keys:
            code = code * 31 + hash(key)
        return code

    def __repr__(self) -> str:
        keys = "-".join(str(k) for k in self.keys)
        return f"{self.text_id}:{keys}"
