import unicodedata
from typing import Optional, Set

from ..corpora.usfm_token import UsfmToken
from .usfm_marker_type import UsfmMarkerType


class TextSegment:
    def __init__(self):
        self._text: GlyphString = GlyphString("")
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
    def text(self) -> "GlyphString":
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
        self._text = GlyphString(self.substring_before(start_index) + replacement + self.substring_after(end_index))
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

        def set_usfm_token(self, token: UsfmToken) -> "TextSegment.Builder":
            self._text_segment._usfm_token = token
            return self

        def set_text(self, text: str) -> "TextSegment.Builder":
            self._text_segment._text = GlyphString(text)
            return self

        def build(self) -> "TextSegment":
            return self._text_segment


class GlyphString:
    def __init__(self, string: str) -> None:
        self._string = string
        self._string_index_by_glyph_index = {
            glyph_index: string_index
            for glyph_index, string_index in enumerate(
                [i for i, c in enumerate(string) if unicodedata.category(c) not in ["Mc", "Mn"]]
            )
        }

    def __len__(self) -> int:
        return len(self._string_index_by_glyph_index)

    def __str__(self):
        return self._string

    def __eq__(self, other) -> bool:
        if not isinstance(other, GlyphString):
            return False
        return self._string == other._string

    def __getitem__(self, key) -> "GlyphString":
        if isinstance(key, int):
            glyph_start = self._normalize_start_index(key)
            glyph_stop = self._normalize_stop_index(glyph_start + 1)
            string_start = self._string_index_by_glyph_index.get(glyph_start, len(self))
            string_stop = self._string_index_by_glyph_index.get(glyph_stop, None)
            return GlyphString(self._string[string_start:string_stop])
        elif isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise TypeError("Steps are not allowed in _glyphString slices")
            glyph_start = self._normalize_start_index(key.start)
            glyph_stop = self._normalize_stop_index(key.stop)
            string_start = self._string_index_by_glyph_index.get(glyph_start, len(self))
            string_stop = self._string_index_by_glyph_index.get(glyph_stop, None)
            return GlyphString(self._string[string_start:string_stop])
        else:
            raise TypeError("Indices must be integers or slices")

    def _normalize_start_index(self, index: Optional[int]) -> int:
        if index is None:
            return 0
        if index < 0:
            return len(self) + index
        return index

    def _normalize_stop_index(self, index: Optional[int]) -> int:
        if index is None:
            return len(self)
        if index < 0:
            return len(self) + index
        return index

    def string_index_to_glyph_index(self, string_index: int) -> int:
        if string_index == len(self._string):
            return len(self)
        for g_index, s_index in self._string_index_by_glyph_index.items():
            if s_index == string_index:
                return g_index
        raise ValueError(f"No corresponding glyph index found for string index {string_index}.")
