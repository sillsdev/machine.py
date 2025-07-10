from typing import Set, Union

from ..usfm_token import UsfmToken
from .usfm_marker_type import UsfmMarkerType


class TextSegment:
    def __init__(self):
        self._text = ""
        self._immediate_preceding_marker: UsfmMarkerType = UsfmMarkerType.NO_MARKER
        self._markers_in_preceding_context: Set[UsfmMarkerType] = set()
        self._previous_segment: Union[TextSegment, None] = None
        self._next_segment: Union[TextSegment, None] = None
        self._index_in_verse: int = 0
        self._num_segments_in_verse: int = 0
        self._usfm_token: Union[UsfmToken, None] = None

    def __eq__(self, value):
        if not isinstance(value, TextSegment):
            return False
        if self._text != value._text:
            return False
        if self._index_in_verse != value._index_in_verse:
            return False
        if self._num_segments_in_verse != value._num_segments_in_verse:
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
    def previous_segment(self) -> Union["TextSegment", None]:
        return self._previous_segment

    @property
    def next_segment(self) -> Union["TextSegment", None]:
        return self._next_segment

    @property
    def length(self) -> int:
        return len(self._text)

    def substring_before(self, index: int) -> str:
        return self._text[:index]

    def substring_after(self, index: int) -> str:
        return self._text[index:]

    def marker_is_in_preceding_context(self, marker: UsfmMarkerType) -> bool:
        return marker in self._markers_in_preceding_context

    def is_first_segment_in_verse(self) -> bool:
        return self._index_in_verse == 0

    def is_last_segment_in_verse(self) -> bool:
        return self._index_in_verse == self._num_segments_in_verse - 1

    def replace_substring(self, start_index: int, end_index: int, replacement: str) -> None:
        self._text = self.substring_before(start_index) + replacement + self.substring_after(end_index)
        if self._usfm_token is not None:
            self._usfm_token.text = self._text

    # These setters need to be implemented outside the builder to avoid circular dependencies
    @previous_segment.setter
    def previous_segment(self, previous_segment: "TextSegment") -> None:
        self._previous_segment = previous_segment

    @next_segment.setter
    def next_segment(self, next_segment: "TextSegment") -> None:
        self._next_segment = next_segment

    def set_index_in_verse(self, index_in_verse: int) -> None:
        self._index_in_verse = index_in_verse

    def set_num_segments_in_verse(self, num_segments_in_verse: int) -> None:
        self._num_segments_in_verse = num_segments_in_verse

    class Builder:
        def __init__(self):
            self._text_segment = TextSegment()

        def set_previous_segment(self, previous_segment: "TextSegment") -> "TextSegment.Builder":
            self._text_segment._previous_segment = previous_segment
            return self

        def add_preceding_marker(self, marker: UsfmMarkerType) -> "TextSegment.Builder":
            self._text_segment._immediate_preceding_marker = marker
            self._text_segment._markers_in_preceding_context.add(marker)
            return self

        def set_usfm_token(self, token: UsfmToken) -> "TextSegment.Builder":
            self._text_segment._usfm_token = token
            return self

        def set_text(self, text: str) -> "TextSegment.Builder":
            self._text_segment._text = text
            return self

        def build(self) -> "TextSegment":
            return self._text_segment
