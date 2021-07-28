from typing import Generator

from .corpora_helpers import gen
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection


class NullTextAlignmentCollection(TextAlignmentCollection):
    def __init__(self, id: str, sort_key: str) -> None:
        self._id = id
        self._sort_key = sort_key

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    @property
    def alignments(self) -> Generator[TextAlignment, None, None]:
        return gen()

    def invert(self) -> "NullTextAlignmentCollection":
        return self
