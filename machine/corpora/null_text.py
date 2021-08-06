from ..utils.context_managed_generator import ContextManagedGenerator
from .corpora_helpers import gen
from .text import Text
from .text_segment import TextSegment


class NullText(Text):
    def __init__(self, id: str, sort_key: str) -> None:
        self._id = id
        self._sort_key = sort_key

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    def get_segments(self, include_text: bool = True) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(gen())
