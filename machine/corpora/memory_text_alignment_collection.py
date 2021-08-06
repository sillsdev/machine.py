from typing import Iterable

from ..utils.context_managed_generator import ContextManagedGenerator
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
    def alignments(self) -> ContextManagedGenerator[TextAlignment, None, None]:
        return ContextManagedGenerator(gen(self._alignments))

    def invert(self) -> "MemoryTextAlignmentCollection":
        return MemoryTextAlignmentCollection(self._id, (ta.invert() for ta in self._alignments))
