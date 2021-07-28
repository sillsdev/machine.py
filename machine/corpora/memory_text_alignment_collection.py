from typing import Generator, Iterable

from .corpora_helpers import gen
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection


class MemoryTextAlignmentCollection(TextAlignmentCollection):
    def __init__(self, id: str, alignments: Iterable[TextAlignment] = []) -> None:
        self._id = id
        self._alignments = list(alignments)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    @property
    def alignments(self) -> Generator[TextAlignment, None, None]:
        return gen(self._alignments)

    def invert(self) -> "MemoryTextAlignmentCollection":
        return MemoryTextAlignmentCollection(self._id, (ta.invert() for ta in self._alignments))
