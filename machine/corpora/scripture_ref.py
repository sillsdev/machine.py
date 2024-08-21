from __future__ import annotations

from functools import total_ordering
from typing import List, Optional

from ..scripture.constants import ENGLISH_VERSIFICATION
from ..scripture.verse_ref import VerseRef, Versification
from ..utils.comparable import Comparable
from .scripture_element import ScriptureElement


@total_ordering
class ScriptureRef(Comparable):
    def __init__(self, ref: Optional[VerseRef] = None, path: Optional[List[ScriptureElement]] = None) -> None:
        self._verse_ref: VerseRef = ref if ref is not None else VerseRef()
        self._path: List[ScriptureElement] = path if path is not None else []

    _empty: Optional[ScriptureRef] = None

    @classmethod
    def parse(cls, selection: str, versification: Optional[Versification] = None) -> ScriptureRef:
        parts: List[str] = selection.split("/")
        if len(parts) == 1:
            return cls(
                VerseRef.from_string(parts[0], versification if versification is not None else ENGLISH_VERSIFICATION)
            )
        vref: str = parts[0]
        path: List[ScriptureElement] = []
        for part in parts[1:]:
            elem: List[str] = part.split(":")
            if len(elem) == 1:
                path.append(ScriptureElement(0, elem[0]))
            else:
                path.append(ScriptureElement(int(elem[0]), elem[1]))

        return cls(
            VerseRef.from_string(vref, versification if versification is not None else ENGLISH_VERSIFICATION), path
        )

    @property
    def verse_ref(self) -> VerseRef:
        return self._verse_ref

    @property
    def path(self) -> List[ScriptureElement]:
        return self._path

    @property
    def book_num(self) -> int:
        return self.verse_ref.book_num

    @property
    def chapter_num(self) -> int:
        return self.verse_ref.chapter_num

    @property
    def verse_num(self) -> int:
        return self.verse_ref.verse_num

    @property
    def book(self) -> str:
        return self.verse_ref.book

    @property
    def chapter(self) -> str:
        return self.verse_ref.chapter

    @property
    def verse(self) -> str:
        return self.verse_ref.verse

    @property
    def versification(self) -> Versification:
        return self.verse_ref.versification

    @property
    def is_empty(self) -> bool:
        return self.verse_ref.is_default

    @property
    def is_verse(self) -> bool:
        return VerseRef.verse_num != 0 and len(self.path) == 0

    def to_relaxed(self) -> ScriptureRef:
        return ScriptureRef(self.verse_ref, [pe.to_relaxed() for pe in self.path])

    def change_versification(self, versification: Versification) -> ScriptureRef:
        vr: VerseRef = self.verse_ref.copy()
        vr.change_versification(versification)
        return ScriptureRef(vr, self.path)

    def compare_to(self, other: object, compare_segments: bool = True) -> int:
        if not isinstance(other, ScriptureRef):
            raise TypeError("other is not a ScriptureRef object.")
        if self is other:
            return 0

        res = self.verse_ref.compare_to(other.verse_ref, compare_segments=compare_segments)
        if res != 0:
            return res

        for se1, se2 in zip(self.path, other.path):
            res = se1.compare_to(se2)
            if res != 0:
                return res
        if len(self.path) < len(other.path):
            return -1
        elif len(self.path) > len(other.path):
            return 1
        return 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScriptureRef):
            return NotImplemented
        return self.verse_ref == other.verse_ref and self.path == other.path

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScriptureRef):
            return NotImplemented
        return self.compare_to(other) < 0

    def __hash__(self) -> int:
        return hash((self.verse_ref, tuple(self.path)))

    def __repr__(self) -> str:
        return f"{self.verse_ref}/{'/'.join(str(se) for se in self.path)}"


EMPTY_SCRIPTURE_REF = ScriptureRef()
