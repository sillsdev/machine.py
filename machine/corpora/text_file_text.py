from pathlib import Path
from typing import Generator, List, Union

from ..utils.string_utils import is_integer
from ..utils.typeshed import StrPath
from .row_ref import RowRef
from .text_base import TextBase
from .text_row import TextRow


class TextFileText(TextBase):
    def __init__(self, id: str, filename: StrPath) -> None:
        super().__init__(id, id)
        self._filename = Path(filename)

    @property
    def filename(self) -> Path:
        return self._filename

    def _get_rows(self) -> Generator[TextRow, None, None]:
        with open(self._filename, "r", encoding="utf-8-sig") as file:
            section_num = 1
            segment_num = 1
            for line in file:
                line = line.rstrip("\r\n")
                if line.lower().startswith("// section"):
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
                    yield self._create_row(line, RowRef(keys))
                    segment_num += 1

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        with open(self._filename, mode="rb") as file:
            return sum(1 for line in file if include_empty or len(line.strip()) > 0)
