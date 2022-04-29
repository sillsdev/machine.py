from pathlib import Path
from typing import Generator

from ..utils.typeshed import StrPath
from .text_base import TextBase
from .text_file_ref import TextFileRef
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
            line_num = 1
            for line in file:
                line = line.rstrip("\r\n")
                yield self._create_row(line, TextFileRef(self.id, line_num))
                line_num += 1

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        with open(self._filename, mode="rb") as file:
            return sum(1 for line in file if include_empty or len(line.strip()) > 0)
