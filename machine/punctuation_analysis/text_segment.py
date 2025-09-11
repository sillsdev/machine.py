import unicodedata
from typing import Optional, Set

from ..corpora.usfm_token import UsfmToken
from .usfm_marker_type import UsfmMarkerType


class TextSegment:
    def __init__(self):
        self._text: GraphemeString = GraphemeString("")
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
    def text(self) -> "GraphemeString":
        return self._text

    @property
    def length(self) -> int:
        return len(self._text)

    def substring_before(self, index: int) -> str:
        return self._text[:index].string

    def substring_after(self, index: int) -> str:
        return self._text[index:].string

    def marker_is_in_preceding_context(self, marker: UsfmMarkerType) -> bool:
        return marker in self._markers_in_preceding_context

    def is_first_segment_in_verse(self) -> bool:
        return self.index_in_verse == 0

    def is_last_segment_in_verse(self) -> bool:
        return self.index_in_verse == self.num_segments_in_verse - 1

    def replace_substring(self, start_index: int, end_index: int, replacement: str) -> None:
        self._text = GraphemeString(self.substring_before(start_index) + replacement + self.substring_after(end_index))
        if self._usfm_token is not None:
            self._usfm_token.text = self._text.string

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

        def set_usfm_token(self, token: UsfmToken) -> "TextSegment.Builder":
            self._text_segment._usfm_token = token
            return self

        def set_text(self, text: str) -> "TextSegment.Builder":
            self._text_segment._text = GraphemeString(text)
            return self

        def build(self) -> "TextSegment":
            return self._text_segment


class GraphemeString:
    def __init__(self, string: str) -> None:
        self._string = string
        self._string_index_by_grapheme_index = {
            grapheme_index: string_index
            for grapheme_index, string_index in enumerate(
                [i for i, c in enumerate(string) if unicodedata.category(c) not in ["Mc", "Mn"]]
            )
        }

    def __len__(self) -> int:
        return len(self._string_index_by_grapheme_index)

    @property
    def string(self) -> str:
        return self._string

    def __str__(self):
        return self._string

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphemeString):
            return False
        return self._string == other.string

    def __getitem__(self, key) -> "GraphemeString":
        if isinstance(key, int):
            grapheme_start = self._normalize_start_index(key)
            grapheme_stop = self._normalize_stop_index(grapheme_start + 1)
            string_start = self._string_index_by_grapheme_index.get(grapheme_start, len(self))
            string_stop = self._string_index_by_grapheme_index.get(grapheme_stop, None)
            return GraphemeString(self._string[string_start:string_stop])
        elif isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise TypeError("Steps are not allowed in _GraphemeString slices")
            grapheme_start = self._normalize_start_index(key.start)
            grapheme_stop = self._normalize_stop_index(key.stop)
            string_start = self._string_index_by_grapheme_index.get(grapheme_start, len(self))
            string_stop = self._string_index_by_grapheme_index.get(grapheme_stop, None)
            return GraphemeString(self._string[string_start:string_stop])
        else:
            raise TypeError("Indices must be integers or slices")

    def _normalize_start_index(self, index: int | None) -> int:
        if index is None:
            return 0
        if index < 0:
            return len(self) + index
        return index

    def _normalize_stop_index(self, index: int | None) -> int:
        if index is None:
            return len(self)
        if index < 0:
            return len(self) + index
        return index

    def string_index_to_grapheme_index(self, string_index: int) -> int:
        if string_index == len(self._string):
            return len(self)
        for g_index, s_index in self._string_index_by_grapheme_index.items():
            if s_index == string_index:
                return g_index
        raise ValueError(f"No corresponding grapheme index found for string index {string_index}.")
