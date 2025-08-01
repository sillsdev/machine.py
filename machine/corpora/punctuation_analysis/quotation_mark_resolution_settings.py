from abc import ABC, abstractmethod
from typing import Set

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_string_match import QuotationMarkStringMatch


class QuotationMarkResolutionSettings(ABC):

    @abstractmethod
    def is_valid_opening_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool: ...

    @abstractmethod
    def is_valid_closing_quotation_mark(self, quotation_mark_match: QuotationMarkStringMatch) -> bool: ...

    @property
    @abstractmethod
    def opening_quotation_mark_regex(self) -> regex.Pattern: ...

    @property
    @abstractmethod
    def closing_quotation_mark_regex(self) -> regex.Pattern: ...

    @abstractmethod
    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool: ...

    @property
    @abstractmethod
    def should_rely_on_paragraph_markers(self) -> bool: ...

    @abstractmethod
    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]: ...

    @abstractmethod
    def metadata_matches_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool: ...
