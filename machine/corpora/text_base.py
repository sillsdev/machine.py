from typing import Any

from .text import Text
from .text_row import TextRow, TextRowFlags
from .text_row_content_type import TextRowContentType


class TextBase(Text):
    def __init__(
        self, id: str, sort_key: str, default_content_type: TextRowContentType = TextRowContentType.SEGMENT
    ) -> None:
        self._id = id
        self._sort_key = sort_key
        self._default_content_type = default_content_type

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    @property
    def default_content_type(self) -> TextRowContentType:
        return self._default_content_type

    def _create_row(self, text: str, ref: Any, flags: TextRowFlags = TextRowFlags.SENTENCE_START) -> TextRow:
        text = text.strip()
        return TextRow(self.id, ref, [text] if len(text) > 0 else [], flags, content_type=self.default_content_type)

    def _create_empty_row(self, ref: Any, flags: TextRowFlags = TextRowFlags.NONE) -> TextRow:
        return TextRow(self.id, ref, flags=flags, content_type=self.default_content_type)
