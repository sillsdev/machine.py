import re
from pathlib import Path
from typing import Generator

from ..utils.string_utils import parse_integer
from ..utils.typeshed import StrPath
from .multi_key_ref import MultiKeyRef
from .text_base import TextBase
from .text_row import TextRow, TextRowFlags


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
                columns = line.split("\t")
                flags = TextRowFlags.SENTENCE_START
                if len(columns) > 1:
                    key_strs = re.split(r"[-_]", columns[0].strip())
                    keys = []
                    for key_str in key_strs:
                        key_int = parse_integer(key_str)
                        keys.append(key_int if key_int is not None else key_str)
                    row_ref = MultiKeyRef(self.id, keys)
                    line = columns[1]
                    if len(columns) == 3:
                        flags = TextRowFlags.NONE
                        for flag_str in columns[2].split(","):
                            flag_str = flag_str.strip().lower()
                            if flag_str in {"sentence_start", "ss"}:
                                flags |= TextRowFlags.SENTENCE_START
                            elif flag_str in {"in_range", "ir"}:
                                flags |= TextRowFlags.IN_RANGE
                            elif flag_str in {"range_start", "rs"}:
                                flags |= TextRowFlags.RANGE_START
                else:
                    row_ref = MultiKeyRef(self.id, [line_num])
                yield self._create_row(line, row_ref, flags)
                line_num += 1

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        with open(self._filename, mode="rb") as file:
            return sum(1 for line in file if include_empty or len(line.strip()) > 0)
