from __future__ import annotations

from functools import total_ordering
from typing import Optional

from ..utils.comparable import Comparable


@total_ordering
class ScriptureElement(Comparable):
    def __init__(self, position: int, name: str) -> None:
        self._position = position
        self._name = name

    @property
    def position(self) -> int:
        return self._position

    @property
    def name(self) -> str:
        return self._name

    def compare_to(self, other: object, strict: Optional[bool] = True) -> int:
        if not isinstance(other, ScriptureElement):
            raise (TypeError("other is not a ScriptureElement object."))
        if self is other:
            return 0

        if strict:
            res = self.position - other.position
            if res != 0:
                return res

        return (self.name > other.name) - (self.name < other.name)

    def __eq__(self, other: ScriptureElement) -> bool:
        if not isinstance(other, ScriptureElement):
            return NotImplemented

        return self.position == other.position and self.name == other.name

    def __lt__(self, other: ScriptureElement) -> bool:
        if not isinstance(other, ScriptureElement):
            return NotImplemented

        return self.compare_to(other) < 0

    def __hash__(self) -> int:
        return hash((self.position, self.name))

    def __repr__(self):
        if self.position == 0:
            return self.name
        return f"{self.position}:{self.name}"