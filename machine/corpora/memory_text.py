from typing import Iterable, Optional

from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_helpers import gen
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

    def get_segments(
        self, include_text: bool = True, based_on: Optional[Text] = None
    ) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(gen(self._segments))
