from typing import Any

from .text import Text
from .text_row import TextRow, TextRowFlags


class TextBase(Text):
    def __init__(self, id: str, sort_key: str) -> None:
        self._id = id
        self._sort_key = sort_key

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    def _create_row(self, text: str, ref: Any, flags: TextRowFlags = TextRowFlags.SENTENCE_START) -> TextRow:
        text = text.strip()
        return TextRow(self.id, ref, [text] if len(text) > 0 else [], flags)

    def _create_empty_row(self, ref: Any, flags: TextRowFlags = TextRowFlags.NONE) -> TextRow:
        return TextRow(self.id, ref, flags=flags)
