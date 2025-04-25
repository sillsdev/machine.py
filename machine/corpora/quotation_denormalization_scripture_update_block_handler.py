from typing import List, Union

from .analysis.depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .analysis.quotation_mark_finder import QuotationMarkFinder
from .analysis.quotation_mark_resolver import QuotationMarkResolver
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .analysis.quote_convention import QuoteConvention
from .analysis.quote_convention_set import QuoteConventionSet
from .analysis.text_segment import TextSegment
from .analysis.usfm_marker_type import UsfmMarkerType
from .basic_quotation_mark_resolver import BasicQuotationMarkResolver
from .quotation_denormalization_action import QuotationDenormalizationAction
from .quotation_denormalization_resolution_settings import QuotationDenormalizationResolutionSettings
from .quotation_denormalization_settings import QuotationDenormalizationSettings
from .scripture_update_block import ScriptureUpdateBlock
from .scripture_update_block_handler import ScriptureUpdateBlockHandler
from .scripture_update_element import ScriptureUpdateElement, ScriptureUpdateElementType
from .usfm_token import UsfmToken, UsfmTokenType


class QuotationDenormalizationScriptureUpdateBlockHandler(ScriptureUpdateBlockHandler):

    def __init__(
        self,
        source_quote_convention: QuoteConvention,
        target_quote_convention: QuoteConvention,
        settings: QuotationDenormalizationSettings = QuotationDenormalizationSettings(),
    ):
        super().__init__()
        self._source_quote_convention: QuoteConvention = source_quote_convention
        self._target_quote_convention: QuoteConvention = target_quote_convention
        self._settings: QuotationDenormalizationSettings = settings

        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([self._source_quote_convention.normalize()])
        )
        self._next_scripture_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

        # Each embed represents a separate context for quotation marks
        # (i.e. you can't open a quote in one context and close it in another)
        # so we need to keep track of the verse and embed contexts separately.
        resolution_settings = QuotationDenormalizationResolutionSettings(
            self._source_quote_convention, self._target_quote_convention
        )
        self._verse_text_quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            resolution_settings
        )
        self._embed_quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            resolution_settings
        )
        self._simple_quotation_mark_resolver: QuotationMarkResolver = BasicQuotationMarkResolver(resolution_settings)
        self._current_denormalization_action = QuotationDenormalizationAction.APPLY_FULL

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        if self._current_denormalization_action is QuotationDenormalizationAction.SKIP:
            return block
        if self._current_denormalization_action is QuotationDenormalizationAction.APPLY_BASIC:
            return self._apply_simple_denormalization(block)
        return self._apply_full_denormalization(block)

    def _apply_simple_denormalization(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        for element in block._elements:
            if element.type == ScriptureUpdateElementType.EMBED_BLOCK or (
                element.type == ScriptureUpdateElementType.EXISTING_TEXT
                and not self._settings.should_run_on_existing_text()
            ):
                continue

            self._process_scripture_element(element, self._simple_quotation_mark_resolver)
        return block

    def _apply_full_denormalization(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        if len(block.elements) > 0 and block.elements[0].type == ScriptureUpdateElementType.EMBED:
            return self._process_embed_block(block)

        return self._process_verse_text_block(block)

    def _process_embed_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        self._embed_quotation_mark_resolver.reset()
        for element in block._elements:
            if (
                element.type == ScriptureUpdateElementType.EXISTING_TEXT
                and not self._settings.should_run_on_existing_text()
            ):
                continue

            self._process_scripture_element(element, self._embed_quotation_mark_resolver)
        return block

    def _process_verse_text_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        for element in block._elements:
            if element.type == ScriptureUpdateElementType.EMBED_BLOCK:
                continue
            if (
                element.type == ScriptureUpdateElementType.EXISTING_TEXT
                and not self._settings.should_run_on_existing_text()
            ):
                continue

            self._process_scripture_element(element, self._verse_text_quotation_mark_resolver)
        return block

    def _process_scripture_element(
        self, element: ScriptureUpdateElement, quotation_mark_resolver: QuotationMarkResolver
    ) -> None:
        text_segments: List[TextSegment] = self._create_text_segments(element)
        quotation_mark_matches: List[QuotationMarkStringMatch] = (
            self._quotation_mark_finder.find_all_potential_quotation_marks_in_text_segments(text_segments)
        )
        for resolved_quotation_mark in quotation_mark_resolver.resolve_quotation_marks(quotation_mark_matches):
            resolved_quotation_mark.update_quotation_mark(self._target_quote_convention)

    def _create_text_segments(self, element: ScriptureUpdateElement) -> List[TextSegment]:
        text_segments: List[TextSegment] = []
        for token in element.get_tokens():
            if token.type == UsfmTokenType.CHAPTER:
                self._start_new_chapter(token)
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.ChapterMarker)
            elif token.type == UsfmTokenType.VERSE:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.VerseMarker)
            elif token.type == UsfmTokenType.PARAGRAPH:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.ParagraphMarker)
            elif token.type == UsfmTokenType.CHARACTER:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.CharacterMarker)
            elif token.type == UsfmTokenType.NOTE:
                self._next_scripture_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)
            elif token.type == UsfmTokenType.TEXT:
                text_segment: Union[TextSegment, None] = self._create_text_segment(token)
                if text_segment is not None:
                    text_segments.append(text_segment)
        return self._set_previous_and_next_for_segments(text_segments)

    def _start_new_chapter(self, token: UsfmToken) -> None:
        chapter_number: Union[int, None] = int(token.data) if token.data is not None else None
        if chapter_number is not None:
            self._current_denormalization_action = self._settings.get_action_for_chapter(chapter_number)
        self._verse_text_quotation_mark_resolver.reset()
        self._next_scripture_text_segment_builder = TextSegment.Builder()

    def _create_text_segment(self, token: UsfmToken) -> Union[TextSegment, None]:
        self._next_scripture_text_segment_builder.set_usfm_token(token)
        if token.text is not None:
            self._next_scripture_text_segment_builder.set_text(token.text)
            text_segment_to_return: TextSegment = self._next_scripture_text_segment_builder.build()
            self._next_scripture_text_segment_builder = TextSegment.Builder()
            return text_segment_to_return
        else:
            self._next_scripture_text_segment_builder = TextSegment.Builder()

    def _set_previous_and_next_for_segments(self, text_segments: List[TextSegment]) -> List[TextSegment]:
        for i in range(len(text_segments)):
            if i > 0:
                text_segments[i].set_previous_segment(text_segments[i - 1])
            if i < len(text_segments) - 1:
                text_segments[i].set_next_segment(text_segments[i + 1])
        return text_segments
