from typing import Iterable

from .text import Text
from .text_segment import TextSegment


class MemoryText(Text):
    def __init__(self, id: str, segments: Iterable[TextSegment] = []) -> None:
        self._id = id
        self._segments = list(segments)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def get_segments(self, include_text: bool = True) -> Iterable[TextSegment]:
        return self._segments
