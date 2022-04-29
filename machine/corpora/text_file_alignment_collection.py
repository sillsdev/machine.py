from typing import Generator

from ..utils.typeshed import StrPath
from .aligned_word_pair import AlignedWordPair
from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow
from .text_file_ref import TextFileRef


class TextFileAlignmentCollection(AlignmentCollection):
    def __init__(self, id: str, filename: StrPath) -> None:
        self._id = id
        self._filename = filename

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def _get_rows(self) -> Generator[AlignmentRow, None, None]:
        with open(self._filename, "r", encoding="utf-8-sig") as file:
            line_num = 1
            for line in file:
                line = line.rstrip("\r\n")
                yield AlignmentRow(self.id, TextFileRef(self.id, line_num), AlignedWordPair.parse(line))
                line_num += 1

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        with open(self._filename, mode="rb") as file:
            return sum(1 for line in file if include_empty or len(line.strip()) > 0)
