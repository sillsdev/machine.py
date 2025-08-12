from collections import defaultdict
from re import Pattern
from typing import Dict, List, Optional, Set, Tuple

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention


class QuoteConventionSet:
    def __init__(self, conventions: List[QuoteConvention]):
        self._conventions = conventions
        self._create_quotation_mark_regexes()
        self._create_quotation_mark_pair_map()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuoteConventionSet):
            return False
        return self._conventions == other._conventions

    def _create_quotation_mark_regexes(self) -> None:
        self._opening_quotation_mark_regex = regex.compile(r"")
        self._closing_quotation_mark_regex = regex.compile(r"")
        self._all_quotation_mark_regex = regex.compile(r"")

        opening_quotation_marks: Set[str] = set()
        closing_quotation_marks: Set[str] = set()

        for convention in self._conventions:
            for depth in range(1, convention.num_levels + 1):
                opening_quotation_mark = convention.get_opening_quotation_mark_at_depth(depth)
                closing_quotation_mark = convention.get_closing_quotation_mark_at_depth(depth)
                opening_quotation_marks.add(opening_quotation_mark)
                closing_quotation_marks.add(closing_quotation_mark)

        all_quotation_marks = opening_quotation_marks.union(closing_quotation_marks)

        if len(all_quotation_marks) > 0:
            self._opening_quotation_mark_regex: Pattern = regex.compile(
                r"[" + "".join(sorted(list(opening_quotation_marks))) + "]"
            )
            self._closing_quotation_mark_regex: Pattern = regex.compile(
                r"[" + "".join(sorted(list(closing_quotation_marks))) + "]"
            )
            self._all_quotation_mark_regex: Pattern = regex.compile(
                r"[" + "".join(sorted(list(all_quotation_marks))) + "]"
            )

    def _create_quotation_mark_pair_map(self) -> None:
        self.closing_marks_by_opening_mark: Dict[str, set[str]] = defaultdict(set)
        self.opening_marks_by_closing_mark: Dict[str, set[str]] = defaultdict(set)
        for convention in self._conventions:
            for depth in range(1, convention.num_levels + 1):
                opening_quotation_mark = convention.get_opening_quotation_mark_at_depth(depth)
                closing_quotation_mark = convention.get_closing_quotation_mark_at_depth(depth)
                self.closing_marks_by_opening_mark[opening_quotation_mark].add(closing_quotation_mark)
                self.opening_marks_by_closing_mark[closing_quotation_mark].add(opening_quotation_mark)

    @property
    def opening_quotation_mark_regex(self) -> Pattern:
        return self._opening_quotation_mark_regex

    @property
    def closing_quotation_mark_regex(self) -> Pattern:
        return self._closing_quotation_mark_regex

    @property
    def quotation_mark_regex(self) -> Pattern:
        return self._all_quotation_mark_regex

    def get_quote_convention_by_name(self, name: str) -> Optional[QuoteConvention]:
        for convention in self._conventions:
            if convention.name == name:
                return convention
        return None

    def get_all_quote_convention_names(self) -> List[str]:
        return sorted([qc._name for qc in self._conventions])

    def get_possible_opening_marks(self) -> list[str]:
        return sorted(list(self.closing_marks_by_opening_mark.keys()))

    def get_possible_closing_marks(self) -> list[str]:
        return sorted(list(self.opening_marks_by_closing_mark.keys()))

    def is_valid_opening_quotation_mark(self, quotation_mark: str) -> bool:
        return quotation_mark in self.closing_marks_by_opening_mark

    def is_valid_closing_quotation_mark(self, quotation_mark: str) -> bool:
        return quotation_mark in self.opening_marks_by_closing_mark

    def marks_are_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
        return (opening_mark in self.closing_marks_by_opening_mark) and (
            closing_mark in self.closing_marks_by_opening_mark[opening_mark]
        )

    def is_quotation_mark_direction_ambiguous(self, quotation_mark: str) -> bool:
        return (
            quotation_mark in self.closing_marks_by_opening_mark
            and quotation_mark in self.closing_marks_by_opening_mark[quotation_mark]
        )

    def get_possible_paired_quotation_marks(self, quotation_mark: str) -> Set[str]:
        paired_quotation_marks: Set[str] = set()
        if quotation_mark in self.closing_marks_by_opening_mark:
            paired_quotation_marks.update(self.closing_marks_by_opening_mark[quotation_mark])
        if quotation_mark in self.opening_marks_by_closing_mark:
            paired_quotation_marks.update(self.opening_marks_by_closing_mark[quotation_mark])
        return paired_quotation_marks

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        depths: Set[int] = set()
        for convention in self._conventions:
            depths.update(convention.get_possible_depths(quotation_mark, direction))
        return depths

    def metadata_matches_quotation_mark(
        self, quotation_mark: str, depth: int, direction: QuotationMarkDirection
    ) -> bool:
        for convention in self._conventions:
            if convention.get_expected_quotation_mark(depth, direction) == quotation_mark:
                return True
        return False

    def filter_to_compatible_quote_conventions(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> "QuoteConventionSet":
        return QuoteConventionSet(
            [
                convention
                for convention in self._conventions
                if convention.is_compatible_with_observed_quotation_marks(
                    opening_quotation_marks, closing_quotation_marks
                )
            ]
        )

    def find_most_similar_convention(
        self, tabulated_quotation_marks: QuotationMarkTabulator
    ) -> Tuple[Optional[QuoteConvention], float]:
        best_similarity: float = float("-inf")
        best_quote_convention: Optional[QuoteConvention] = None
        for quote_convention in self._conventions:
            similarity = tabulated_quotation_marks.calculate_similarity(quote_convention)
            if similarity > best_similarity:
                best_similarity = similarity
                best_quote_convention = quote_convention

        return (best_quote_convention, best_similarity)
