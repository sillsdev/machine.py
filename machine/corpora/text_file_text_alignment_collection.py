from typing import Generator

from ..utils.context_managed_generator import ContextManagedGenerator
from ..utils.string_utils import is_integer
from ..utils.typeshed import StrPath
from .aligned_word_pair import AlignedWordPair
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection
from .text_segment_ref import TextSegmentRef


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
    def alignments(self) -> ContextManagedGenerator[TextAlignment, None, None]:
        return ContextManagedGenerator(self._get_alignments())

    def invert(self) -> "TextFileTextAlignmentCollection":
        return TextFileTextAlignmentCollection(self._id, self._filename, not self._invert)

    def _get_alignments(self) -> Generator[TextAlignment, None, None]:
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
                    yield TextAlignment(
                        self._id, TextSegmentRef(section_num, segment_num), AlignedWordPair.parse(line, self._invert)
                    )
                    segment_num += 1
