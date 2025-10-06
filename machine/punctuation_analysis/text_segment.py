from typing import Optional, Set

from ..corpora.usfm_token import UsfmToken
from .usfm_marker_type import UsfmMarkerType


class TextSegment:
    def __init__(self):
        self._text = ""
        self.book: Optional[str] = None
        self.chapter: Optional[int] = None
        self._immediate_preceding_marker: UsfmMarkerType = UsfmMarkerType.NO_MARKER
        self._markers_in_preceding_context: Set[UsfmMarkerType] = set()
        self.previous_segment: Optional[TextSegment] = None
        self.next_segment: Optional[TextSegment] = None
        self.index_in_verse: int = 0
        self.num_segments_in_verse: int = 0
        self._usfm_token: Optional[UsfmToken] = None

    def __eq__(self, value):
        if not isinstance(value, TextSegment):
            return False
        if self._text != value._text:
            return False
        if self.index_in_verse != value.index_in_verse:
            return False
        if self.num_segments_in_verse != value.num_segments_in_verse:
            return False
        if self._usfm_token != value._usfm_token:
            return False
        if self._immediate_preceding_marker != value._immediate_preceding_marker:
            return False
        return True

    @property
    def text(self) -> str:
        return self._text

    @property
    def length(self) -> int:
        return len(self._text)

    def substring_before(self, index: int) -> str:
        return str(self._text[:index])

    def substring_after(self, index: int) -> str:
        return str(self._text[index:])

    def marker_is_in_preceding_context(self, marker: UsfmMarkerType) -> bool:
        return marker in self._markers_in_preceding_context

    def is_first_segment_in_verse(self) -> bool:
        return self.index_in_verse == 0

    def is_last_segment_in_verse(self) -> bool:
        return self.index_in_verse == self.num_segments_in_verse - 1

    def replace_substring(self, start_index: int, end_index: int, replacement: str) -> None:
        self._text = self.substring_before(start_index) + replacement + self.substring_after(end_index)
        if self._usfm_token is not None:
            self._usfm_token.text = str(self._text)

    class Builder:
        def __init__(self):
            self._text_segment = TextSegment()

        def set_previous_segment(self, previous_segment: "TextSegment") -> "TextSegment.Builder":
            self._text_segment.previous_segment = previous_segment
            return self

        def add_preceding_marker(self, marker: UsfmMarkerType) -> "TextSegment.Builder":
            self._text_segment._immediate_preceding_marker = marker
            self._text_segment._markers_in_preceding_context.add(marker)
            return self

        def set_book(self, code: str) -> "TextSegment.Builder":
            self._text_segment.book = code
            return self

        def set_chapter(self, number: int) -> "TextSegment.Builder":
            self._text_segment.chapter = number
            return self

        def set_usfm_token(self, token: UsfmToken) -> "TextSegment.Builder":
            self._text_segment._usfm_token = token
            return self

        def set_text(self, text: str) -> "TextSegment.Builder":
            self._text_segment._text = text
            return self

        def build(self) -> "TextSegment":
            return self._text_segment
