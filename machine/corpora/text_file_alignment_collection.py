import re
from typing import Generator

from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .aligned_word_pair import AlignedWordPair
from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow
from .multi_key_ref import MultiKeyRef


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
                index = line.find("\t")
                if index >= 0:
                    key_strs = re.split(r"[-_]", line[:index].strip())
                    keys = []
                    for key_str in key_strs:
                        key_int = parse_integer(key_str)
                        keys.append(key_int if key_int is not None else key_str)
                    row_ref = MultiKeyRef(self.id, keys)
                    line = line[index + 1 :]
                else:
                    row_ref = MultiKeyRef(self.id, [line_num])
                yield AlignmentRow(self.id, row_ref, AlignedWordPair.from_string(line))
                line_num += 1

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        with open(self._filename, mode="rb") as file:
            return sum(1 for line in file if include_empty or len(line.strip()) > 0)
