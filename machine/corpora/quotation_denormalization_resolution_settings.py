from typing import Set

from .analysis.quotation_mark_direction import QuotationMarkDirection
from .analysis.quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .analysis.quote_convention import QuoteConvention
from .analysis.quote_convention_set import QuoteConventionSet


class QuotationDenormalizationResolutionSettings(QuotationMarkResolutionSettings):
    def __init__(self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention):
        self._normalized_source_quote_convention = source_quote_convention.normalize()
        self._normalized_quote_convention_singleton_set = QuoteConventionSet([self._normalized_source_quote_convention])
        self._target_quote_convention = target_quote_convention

    def is_valid_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_opening_quotation_mark(self._normalized_quote_convention_singleton_set)

    def is_valid_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_closing_quotation_mark(self._normalized_quote_convention_singleton_set)

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
        return self._normalized_quote_convention_singleton_set.are_marks_a_valid_pair(opening_mark, closing_mark)

    def should_rely_on_paragraph_markers(self):
        return False

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        return self._normalized_source_quote_convention.get_possible_depths(quotation_mark, direction)

    def does_metadata_match_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool:
        return self._normalized_source_quote_convention.get_expected_quotation_mark(depth, direction) == quotation_mark
