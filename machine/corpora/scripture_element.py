from __future__ import annotations

from functools import total_ordering

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

    def to_relaxed(self) -> ScriptureElement:
        return ScriptureElement(0, self.name)

    def compare_to(self, other: object) -> int:
        if not isinstance(other, ScriptureElement):
            raise (TypeError("other is not a ScriptureElement object."))
        if self is other:
            return 0

        if self.position == 0 or other.position == 0:
            if self.name == other.name:
                return 0
            # position 0 is always greater than any other position
            if self.position == 0 and other.position != 0:
                return 1
            if other.position == 0 and self.position != 0:
                return -1
            return (self.name > other.name) - (self.name < other.name)
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
