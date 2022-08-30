from typing import Any

from .text import Text
from .text_row import TextRow


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

    def _create_row(
        self,
        text: str,
        ref: Any,
        is_sentence_start: bool = True,
        is_in_range: bool = False,
        is_range_start: bool = False,
    ) -> TextRow:
        text = text.strip()
        return TextRow(
            self.id,
            ref,
            [text] if len(text) > 0 else [],
            is_sentence_start,
            is_in_range,
            is_range_start,
            is_empty=len(text) == 0,
        )

    def _create_empty_row(self, ref: Any, is_in_range: bool = False) -> TextRow:
        return TextRow(self.id, ref, is_in_range=is_in_range)
