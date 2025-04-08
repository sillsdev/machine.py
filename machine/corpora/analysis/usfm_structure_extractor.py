from typing import Optional, Sequence

from ..usfm_parser_handler import UsfmParserHandler
from ..usfm_parser_state import UsfmParserState
from ..usfm_token import UsfmAttribute
from .chapter import Chapter
from .text_segment import TextSegment
from .usfm_marker_type import UsfmMarkerType
from .verse import Verse


class UsfmStructureExtractor(UsfmParserHandler):
    def __init__(self):
        self._reset()

    def _reset(self):
        self.text_segments: list[TextSegment] = []
        self.next_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.ChapterMarker)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.ParagraphMarker)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CharacterMarker)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CharacterMarker)

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.VerseMarker)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def end_table(self, state: UsfmParserState) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def text(self, state: UsfmParserState, text: str) -> None:
        if not state.is_verse_text:
            return
        if len(text) > 0:
            self.next_text_segment_builder.set_text(text)
            text_segment: TextSegment = self.next_text_segment_builder.build()
            if len(self.text_segments) > 0:
                self.text_segments[-1].set_next_segment(text_segment)
            self.text_segments.append(text_segment)
        self.next_text_segment_builder = TextSegment.Builder()

    def get_chapters(self) -> list[Chapter]:
        chapters: list[Chapter] = []
        current_chapter_verses: list[Verse] = []
        current_verse_segments: list[TextSegment] = []
        for text_segment in self.text_segments:
            if text_segment.is_marker_in_preceding_context(UsfmMarkerType.VerseMarker):
                if len(current_verse_segments) > 0:
                    current_chapter_verses.append(Verse(current_verse_segments))
                current_verse_segments = []
            if text_segment.is_marker_in_preceding_context(UsfmMarkerType.ChapterMarker):
                if len(current_chapter_verses) > 0:
                    chapters.append(Chapter(current_chapter_verses))
                current_chapter_verses = []
            current_verse_segments.append(text_segment)
        if len(current_verse_segments) > 0:
            current_chapter_verses.append(Verse(current_verse_segments))
        if len(current_chapter_verses) > 0:
            chapters.append(Chapter(current_chapter_verses))
        return chapters
