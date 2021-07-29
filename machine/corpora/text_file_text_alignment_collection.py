from typing import Generator

from machine.corpora.aligned_word_pair import AlignedWordPair
from machine.corpora.text_segment_ref import TextSegmentRef

from ..string_utils import is_integer
from ..typeshed import StrPath
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection


class TextFileTextAlignmentCollection(TextAlignmentCollection):
    def __init__(self, id: str, filename: StrPath, invert: bool = False) -> None:
        self._id = id
        self._filename = filename
        self._invert = invert

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    @property
    def alignments(self) -> Generator[TextAlignment, None, None]:
        with open(self._filename, "r", encoding="utf-8") as file:
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
                    yield TextAlignment(
                        self._id, TextSegmentRef(section_num, segment_num), AlignedWordPair.parse(line, self._invert)
                    )
                    segment_num += 1

    def invert(self) -> "TextFileTextAlignmentCollection":
        return TextFileTextAlignmentCollection(self._id, self._filename, not self._invert)
