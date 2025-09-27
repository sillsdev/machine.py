from typing import List, Optional

from ..corpora.usfm_token import UsfmToken, UsfmTokenType
from ..corpora.usfm_update_block import UsfmUpdateBlock
from ..corpora.usfm_update_block_element import UsfmUpdateBlockElement, UsfmUpdateBlockElementType
from ..corpora.usfm_update_block_handler import UsfmUpdateBlockHandler
from ..punctuation_analysis.depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from ..punctuation_analysis.quotation_mark_finder import QuotationMarkFinder
from ..punctuation_analysis.quotation_mark_metadata import QuotationMarkMetadata
from ..punctuation_analysis.quotation_mark_resolver import QuotationMarkResolver
from ..punctuation_analysis.quotation_mark_string_match import QuotationMarkStringMatch
from ..punctuation_analysis.quote_convention import QuoteConvention
from ..punctuation_analysis.quote_convention_set import QuoteConventionSet
from ..punctuation_analysis.text_segment import TextSegment
from ..punctuation_analysis.usfm_marker_type import UsfmMarkerType
from .fallback_quotation_mark_resolver import FallbackQuotationMarkResolver
from .quotation_mark_update_resolution_settings import QuotationMarkUpdateResolutionSettings
from .quotation_mark_update_settings import QuotationMarkUpdateSettings
from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy


class QuoteConventionChangingUsfmUpdateBlockHandler(UsfmUpdateBlockHandler):

    def __init__(
        self,
        old_quote_convention: QuoteConvention,
        new_quote_convention: QuoteConvention,
        settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
    ):
        super().__init__()
        self._old_quote_convention: QuoteConvention = old_quote_convention
        self._new_quote_convention: QuoteConvention = new_quote_convention
        self._settings: QuotationMarkUpdateSettings = settings

        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([self._old_quote_convention])
        )
        self._next_scripture_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

        resolution_settings = QuotationMarkUpdateResolutionSettings(self._old_quote_convention)

        # Each embed represents a separate context for quotation marks
        # (i.e. you can't open a quote in one context and close it in another)
        # so we need to keep track of the verse and embed contexts separately.
        self._verse_text_quotation_mark_resolver: DepthBasedQuotationMarkResolver = DepthBasedQuotationMarkResolver(
            resolution_settings
        )
        self._embed_quotation_mark_resolver: DepthBasedQuotationMarkResolver = DepthBasedQuotationMarkResolver(
            resolution_settings
        )
        self._simple_quotation_mark_resolver: FallbackQuotationMarkResolver = FallbackQuotationMarkResolver(
            resolution_settings
        )
        self._current_strategy = QuotationMarkUpdateStrategy.APPLY_FULL
        self._current_chapter_number: int = 0
        self._current_verse_number: int = 0

    def process_block(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock:
        self._check_for_chapter_change(block)
        self._check_for_verse_change(block)
        if self._current_strategy is QuotationMarkUpdateStrategy.SKIP:
            return block
        if self._current_strategy is QuotationMarkUpdateStrategy.APPLY_FALLBACK:
            return self._apply_fallback_updating(block)
        return self._apply_standard_updating(block)

    def _apply_fallback_updating(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock:
        for element in block.elements:
            self._process_scripture_element(element, self._simple_quotation_mark_resolver)
        return block

    def _apply_standard_updating(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock:
        for element in block.elements:
            if element.type == UsfmUpdateBlockElementType.EMBED:
                self._embed_quotation_mark_resolver.reset()
                self._process_scripture_element(element, self._embed_quotation_mark_resolver)
            else:
                self._process_scripture_element(element, self._verse_text_quotation_mark_resolver)

        return block

    def _process_scripture_element(
        self, element: UsfmUpdateBlockElement, quotation_mark_resolver: QuotationMarkResolver
    ) -> None:
        text_segments: List[TextSegment] = self._create_text_segments(element)
        quotation_mark_matches: List[QuotationMarkStringMatch] = (
            self._quotation_mark_finder.find_all_potential_quotation_marks_in_text_segments(text_segments)
        )
        resolved_quotation_mark_matches: List[QuotationMarkMetadata] = list(
            quotation_mark_resolver.resolve_quotation_marks(quotation_mark_matches)
        )
        self._update_quotation_marks(resolved_quotation_mark_matches)

    def _create_text_segments(self, element: UsfmUpdateBlockElement) -> List[TextSegment]:
        text_segments: List[TextSegment] = []
        for token in element.get_tokens():
            if token.type == UsfmTokenType.VERSE:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.VERSE)
            elif token.type == UsfmTokenType.PARAGRAPH:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.PARAGRAPH)
            elif token.type == UsfmTokenType.CHARACTER:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.CHARACTER)
            elif token.type == UsfmTokenType.NOTE:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.EMBED)
            elif token.type == UsfmTokenType.TEXT:
                text_segment: Optional[TextSegment] = self._create_text_segment(token)
                if text_segment is not None:
                    text_segments.append(text_segment)
        return self._set_previous_and_next_for_segments(text_segments)

    def _create_text_segment(self, token: UsfmToken) -> Optional[TextSegment]:
        self._next_scripture_text_segment_builder.set_usfm_token(token)
        text_segment_to_return: Optional[TextSegment] = None
        if token.text is not None:
            self._next_scripture_text_segment_builder.set_text(token.text)
            text_segment_to_return = self._next_scripture_text_segment_builder.build()
        self._next_scripture_text_segment_builder = TextSegment.Builder()
        return text_segment_to_return

    def _set_previous_and_next_for_segments(self, text_segments: List[TextSegment]) -> List[TextSegment]:
        for i in range(len(text_segments)):
            if i > 0:
                text_segments[i].previous_segment = text_segments[i - 1]
            if i < len(text_segments) - 1:
                text_segments[i].next_segment = text_segments[i + 1]
        return text_segments

    def _update_quotation_marks(self, resolved_quotation_mark_matches: List[QuotationMarkMetadata]) -> None:
        for quotation_mark_index, resolved_quotation_mark_match in enumerate(resolved_quotation_mark_matches):
            previous_length: int = resolved_quotation_mark_match.length
            resolved_quotation_mark_match.update_quotation_mark(self._new_quote_convention)
            updated_length: int = resolved_quotation_mark_match.length

            if previous_length != updated_length:
                self._shift_quotation_mark_metadata_indices(
                    resolved_quotation_mark_matches[quotation_mark_index + 1 :], updated_length - previous_length
                )

    def _shift_quotation_mark_metadata_indices(
        self, quotation_mark_metadata_list: List[QuotationMarkMetadata], shift_amount: int
    ) -> None:
        for quotation_mark_metadata in quotation_mark_metadata_list:
            quotation_mark_metadata.shift_indices(shift_amount)

    def _check_for_chapter_change(self, block: UsfmUpdateBlock) -> None:
        for scripture_ref in block.refs:
            if scripture_ref.chapter_num != self._current_chapter_number:
                self._start_new_chapter(scripture_ref.chapter_num)

    def _start_new_chapter(self, new_chapter_number: int) -> None:
        self._current_chapter_number = new_chapter_number
        self._current_strategy = self._settings.get_action_for_chapter(new_chapter_number)
        self._verse_text_quotation_mark_resolver.reset()
        self._next_scripture_text_segment_builder = TextSegment.Builder()
        self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.CHAPTER)

    def _check_for_verse_change(self, block: UsfmUpdateBlock) -> None:
        for scripture_ref in block.refs:
            if (
                scripture_ref.chapter_num == self._current_chapter_number
                and scripture_ref.verse_num != self._current_verse_number
            ):
                self._start_new_verse(scripture_ref.verse_num)

    def _start_new_verse(self, new_verse_number: int) -> None:
        self._current_verse_number = new_verse_number
        self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.VERSE)
