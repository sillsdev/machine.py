from typing import Set, Union

from ..usfm_token import UsfmToken
from .usfm_marker_type import UsfmMarkerType


class TextSegment:
    def __init__(self):
        self.text = ""
        self.immediate_preceding_marker: UsfmMarkerType = UsfmMarkerType.NoMarker
        self.markers_in_preceding_context: Set[UsfmMarkerType] = set()
        self.previous_segment: Union[TextSegment, None] = None
        self.next_segment: Union[TextSegment, None] = None
        self.index_in_verse: int = 0
        self.num_segments_in_verse: int = 0
        self.usfm_token: Union[UsfmToken, None] = None

    def __eq__(self, value):
        if not isinstance(value, TextSegment):
            return False
        if self.text != value.text:
            return False
        if self.index_in_verse != value.index_in_verse:
            return False
        if self.usfm_token != value.usfm_token:
            return False
        if self.immediate_preceding_marker != value.immediate_preceding_marker:
            return False
        return True

    def get_text(self) -> str:
        return self.text

    def length(self) -> int:
        return len(self.text)

    def substring_before(self, index: int) -> str:
        return self.text[0:index]

    def substring_after(self, index: int) -> str:
        return self.text[index:-1]

    def get_immediate_preceding_marker_type(self) -> UsfmMarkerType:
        return self.immediate_preceding_marker

    def is_marker_in_preceding_context(self, marker: UsfmMarkerType) -> bool:
        return marker in self.markers_in_preceding_context

    def get_previous_segment(self) -> Union["TextSegment", None]:
        return self.previous_segment

    def get_next_segment(self) -> Union["TextSegment", None]:
        return self.next_segment

    def is_first_segment_in_verse(self) -> bool:
        return self.index_in_verse == 0

    def is_last_segment_in_verse(self) -> bool:
        return self.index_in_verse == self.num_segments_in_verse - 1

    def replace_substring(self, start_index: int, end_index: int, replacement: str) -> None:
        self.text = self.text[:start_index] + replacement + self.text[end_index:]
        if self.usfm_token is not None:
            self.usfm_token.text = self.text

    # These setters need to be implemented outside the builder to avoid circular dependencies
    def set_previous_segment(self, previous_segment: "TextSegment") -> None:
        self.previous_segment = previous_segment

    def set_next_segment(self, next_segment: "TextSegment") -> None:
        self.next_segment = next_segment

    def set_index_in_verse(self, index_in_verse: int) -> None:
        self.index_in_verse = index_in_verse

    def set_num_segments_in_verse(self, num_segments_in_verse: int) -> None:
        self.num_segments_in_verse = num_segments_in_verse

    class Builder:
        def __init__(self):
            self.text_segment = TextSegment()

        def set_previous_segment(self, previous_segment: "TextSegment") -> "TextSegment.Builder":
            self.text_segment.previous_segment = previous_segment
            return self

        def add_preceding_marker(self, marker: UsfmMarkerType) -> "TextSegment.Builder":
            self.text_segment.immediate_preceding_marker = marker
            self.text_segment.markers_in_preceding_context.add(marker)
            return self

        def set_usfm_token(self, token: UsfmToken) -> "TextSegment.Builder":
            self.text_segment.usfm_token = token
            return self

        def set_text(self, text: str) -> "TextSegment.Builder":
            self.text_segment.text = text
            return self

        def build(self) -> "TextSegment":
            return self.text_segment
