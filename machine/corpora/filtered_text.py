from contextlib import closing
from typing import Callable, Generator

from .text import Text
from .text_segment import TextSegment


class FilteredText(Text):
    def __init__(self, text: Text, filter: Callable[[TextSegment], bool]) -> None:
        self._text = text
        self._filter = filter

    @property
    def id(self) -> str:
        return self._text.id

    @property
    def sort_key(self) -> str:
        return self._text.sort_key

    def get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
        with closing(self._text.get_segments(include_text)) as segments:
            for segment in segments:
                if self._filter(segment):
                    yield segment
