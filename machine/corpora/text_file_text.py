from pathlib import Path
from typing import Generator

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
                    yield self._create_row(line, RowRef(self.id, section_num, segment_num))
                    segment_num += 1
