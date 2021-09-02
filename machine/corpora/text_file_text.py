from pathlib import Path
from typing import Generator

from ..tokenization.tokenizer import Tokenizer
from ..utils.string_utils import is_integer
from ..utils.typeshed import StrPath
from .text_base import TextBase
from .text_segment import TextSegment
from .text_segment_ref import TextSegmentRef


class TextFileText(TextBase):
    def __init__(self, word_tokenizer: Tokenizer[str, int, str], id: str, filename: StrPath) -> None:
        super().__init__(word_tokenizer, id, id)
        self._filename = Path(filename)

    @property
    def filename(self) -> Path:
        return self._filename

    def _get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
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
                    yield self._create_text_segment(include_text, line, TextSegmentRef(section_num, segment_num))
                    segment_num += 1
