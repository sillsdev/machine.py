from typing import List

from .analysis.quotation_mark_finder import QuotationMarkFinder
from .analysis.quotation_mark_resolver import QuotationMarkResolver
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .analysis.quote_convention import QuoteConvention
from .analysis.quote_convention_set import QuoteConventionSet
from .analysis.text_segment import TextSegment
from .analysis.usfm_marker_type import UsfmMarkerType
from .scripture_update_block import ScriptureUpdateBlock
from .scripture_update_block_handler import ScriptureUpdateBlockHandler
from .scripture_update_element import ScriptureUpdateElement, ScriptureUpdateElementType
from .usfm_token import UsfmTokenType


class QuotationDenormalizationScriptureUpdateBlockHandler(ScriptureUpdateBlockHandler):

    def __init__(self, target_quote_convention: QuoteConvention, should_run_on_existing_text: bool = False):
        super().__init__()
        self._target_quote_convention: QuoteConvention = target_quote_convention
        self._normalized_quote_convention: QuoteConvention = target_quote_convention.normalize()
        self._should_run_on_existing_text: bool = should_run_on_existing_text

        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([self._normalized_quote_convention])
        )
        self._next_scripture_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

        # Each embed represents a separate context for quotation marks
        # (i.e. you can't open a quote in one and close it in another)
        # so we need to keep track of the verse and embed contexts separately.
        self._verse_text_quotation_mark_resolver: QuotationMarkResolver = QuotationMarkResolver(
            QuoteConventionSet([self._normalized_quote_convention])
        )
        self._embed_quotation_mark_resolver: QuotationMarkResolver = QuotationMarkResolver(
            QuoteConventionSet([self._normalized_quote_convention])
        )

    def process_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        if len(block.elements) > 0 and block.elements[0].type == ScriptureUpdateElementType.EMBED:
            return self._process_embed_block(block)

        return self._process_verse_text_block(block)

    def _process_embed_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        self._embed_quotation_mark_resolver.reset()
        for element in block._elements:
            if element.type == ScriptureUpdateElementType.EXISTING_TEXT and not self._should_run_on_existing_text:
                continue

            self._process_scripture_element(element, self._embed_quotation_mark_resolver)
        return block

    def _process_verse_text_block(self, block: ScriptureUpdateBlock) -> ScriptureUpdateBlock:
        for element in block._elements:
            if element.type == ScriptureUpdateElementType.EMBED_BLOCK:
                continue
            if element.type == ScriptureUpdateElementType.EXISTING_TEXT and not self._should_run_on_existing_text:
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
                self._verse_text_quotation_mark_resolver.reset()
                self._next_scripture_text_segment_builder = TextSegment.Builder()
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
                self._next_scripture_text_segment_builder.set_usfm_token(token)
                if token.text is not None:
                    self._next_scripture_text_segment_builder.set_text(token.text)
                    text_segments.append(self._next_scripture_text_segment_builder.build())
                else:
                    self._next_scripture_text_segment_builder = TextSegment.Builder()
        return self._set_previous_and_next_for_segments(text_segments)

    def _set_previous_and_next_for_segments(self, text_segments: List[TextSegment]) -> List[TextSegment]:
        for i in range(len(text_segments)):
            if i > 0:
                text_segments[i].set_previous_segment(text_segments[i - 1])
            if i < len(text_segments) - 1:
                text_segments[i].set_next_segment(text_segments[i + 1])
        return text_segments
