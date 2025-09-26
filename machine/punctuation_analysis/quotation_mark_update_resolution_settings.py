from typing import Set

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention import QuoteConvention
from .quote_convention_set import QuoteConventionSet


class QuotationMarkUpdateResolutionSettings(QuotationMarkResolutionSettings):
    def __init__(self, old_quote_convention: QuoteConvention):
        self._old_quote_convention = old_quote_convention
        self._quote_convention_singleton_set = QuoteConventionSet([self._old_quote_convention])

    def is_valid_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_opening_quotation_mark(self._quote_convention_singleton_set)

    def is_valid_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_closing_quotation_mark(self._quote_convention_singleton_set)

    @property
    def opening_quotation_mark_regex(self) -> regex.Pattern:
        return self._quote_convention_singleton_set.opening_quotation_mark_regex

    @property
    def closing_quotation_mark_regex(self) -> regex.Pattern:
        return self._quote_convention_singleton_set.closing_quotation_mark_regex

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
        return self._quote_convention_singleton_set.marks_are_a_valid_pair(opening_mark, closing_mark)

    @property
    def should_rely_on_paragraph_markers(self):
        return False

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        return self._old_quote_convention.get_possible_depths(quotation_mark, direction)

    def metadata_matches_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool:
        return self._old_quote_convention.get_expected_quotation_mark(depth, direction) == quotation_mark
