from typing import Dict, List, Optional, Sequence

from ..corpora.usfm_parser_handler import UsfmParserHandler
from ..corpora.usfm_parser_state import UsfmParserState
from ..corpora.usfm_token import UsfmAttribute
from ..scripture.canon import book_id_to_number
from .chapter import Chapter
from .text_segment import TextSegment
from .usfm_marker_type import UsfmMarkerType
from .verse import Verse


class UsfmStructureExtractor(UsfmParserHandler):
    def __init__(self):
        self._text_segments: list[TextSegment] = []
        self._next_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._next_text_segment_builder.set_book(code)

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CHAPTER)
        self._next_text_segment_builder.set_chapter(state.verse_ref.chapter_num)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.PARAGRAPH)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CHARACTER)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CHARACTER)

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.VERSE)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED)

    def end_table(self, state: UsfmParserState) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self._next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED)

    def text(self, state: UsfmParserState, text: str) -> None:
        if not state.is_verse_text:
            return
        if len(text) > 0:
            self._next_text_segment_builder.set_text(text)
            text_segment: TextSegment = self._next_text_segment_builder.build()
            # Don't look past verse boundaries, to enable identical functionality in the
            # online one-verse-at-a-time (QuotationMarkDenormalizationScriptureUpdateBlockHandler)
            # and offline whole-book-at-once settings (QuoteConventionDetector)
            if len(self._text_segments) > 0 and not text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE):
                self._text_segments[-1].next_segment = text_segment
                text_segment.previous_segment = self._text_segments[-1]
            self._text_segments.append(text_segment)
        self._next_text_segment_builder = TextSegment.Builder()

    def get_chapters(self, include_chapters: Optional[Dict[int, List[int]]] = None) -> list[Chapter]:
        chapters: list[Chapter] = []
        current_book: int = 0
        current_chapter: int = 0
        current_chapter_verses: list[Verse] = []
        current_verse_segments: list[TextSegment] = []
        for text_segment in self._text_segments:
            if text_segment.book is not None:
                current_book = book_id_to_number(text_segment.book)
            if text_segment.chapter is not None:
                current_chapter = text_segment.chapter
            if include_chapters is not None and current_book > 0:
                if current_book not in include_chapters:
                    continue
                elif (
                    current_chapter > 0
                    and len(include_chapters[current_book]) > 0
                    and current_chapter not in include_chapters[current_book]
                ):
                    continue
            if text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE):
                if len(current_verse_segments) > 0:
                    current_chapter_verses.append(Verse(current_verse_segments))
                current_verse_segments = []
            if text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHAPTER):
                if len(current_chapter_verses) > 0:
                    chapters.append(Chapter(current_chapter_verses))
                current_chapter_verses = []
            current_verse_segments.append(text_segment)
        if len(current_verse_segments) > 0:
            current_chapter_verses.append(Verse(current_verse_segments))
        if len(current_chapter_verses) > 0:
            chapters.append(Chapter(current_chapter_verses))
        return chapters
