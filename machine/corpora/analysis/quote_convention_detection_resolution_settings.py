from typing import Set

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_resolution_settings import QuotationMarkResolutionSettings
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention_set import QuoteConventionSet


class QuoteConventionDetectionResolutionSettings(QuotationMarkResolutionSettings):

    def __init__(self, quote_convention_set: QuoteConventionSet):
        self._quote_convention_set = quote_convention_set

    def is_valid_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_opening_quotation_mark(self._quote_convention_set)

    def is_valid_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool:
        return quotation_mark_match.is_valid_closing_quotation_mark(self._quote_convention_set)

    def get_opening_quotation_mark_regex(self) -> regex.Pattern:
        return self._quote_convention_set.get_opening_quotation_mark_regex()

    def get_closing_quotation_mark_regex(self) -> regex.Pattern:
        return self._quote_convention_set.get_closing_quotation_mark_regex()

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
        return self._quote_convention_set.are_marks_a_valid_pair(opening_mark, closing_mark)

    def should_rely_on_paragraph_markers(self):
        return True

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        return self._quote_convention_set.get_possible_depths(quotation_mark, direction)

    def does_metadata_match_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool:
        return self._quote_convention_set.does_metadata_match_quotation_mark(quotation_mark, depth, direction)
