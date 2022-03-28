from typing import Generator, Iterable

from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow
from .corpora_utils import gen


class MemoryAlignmentCollection(AlignmentCollection):
    def __init__(self, id: str, alignments: Iterable[AlignmentRow] = []) -> None:
        self._id = id
        self._alignments = list(alignments)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        return gen(self._alignments)
