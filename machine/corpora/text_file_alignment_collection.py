from typing import Generator, List, Union

from ..utils.string_utils import is_integer
from ..utils.typeshed import StrPath
from .aligned_word_pair import AlignedWordPair
from .alignment_collection import AlignmentCollection
from .alignment_row import AlignmentRow
from .row_ref import RowRef


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
            section_num = 1
            segment_num = 1
            for line in file:
                line = line.rstrip("\r\n")
                if line.startswith("// section "):
                    section_num_str = line[11:].strip()
                    if is_integer(section_num_str):
                        section_num = int(section_num_str)
                        segment_num = 1
                else:
                    keys: List[Union[str, int]] = []
                    if self.id != "*all*":
                        keys.append(self.id)
                    keys.append(section_num)
                    keys.append(segment_num)
                    yield AlignmentRow(RowRef(keys), AlignedWordPair.parse(line))
                    segment_num += 1
