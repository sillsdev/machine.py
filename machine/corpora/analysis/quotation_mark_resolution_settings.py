from abc import ABC
from typing import Set

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_string_match import QuotationMarkStringMatch


class QuotationMarkResolutionSettings(ABC):

    def is_valid_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool: ...

    def is_valid_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool: ...

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool: ...

    def should_rely_on_paragraph_markers(self) -> bool: ...

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]: ...

    def does_metadata_match_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool: ...
