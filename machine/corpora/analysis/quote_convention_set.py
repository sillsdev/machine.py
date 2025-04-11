from re import Pattern
from typing import Dict, List, Set, Tuple, Union

import regex

from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention


class QuoteConventionSet:
    def __init__(self, conventions: List[QuoteConvention]):
        self.conventions = conventions
        self._create_quote_regexes()
        self._create_quotation_mark_pair_map()

    def _create_quote_regexes(self) -> None:
        opening_quotation_marks: Set[str] = set()
        closing_quotation_marks: Set[str] = set()
        all_quotation_marks: Set[str] = set()

        if len(self.conventions) > 0:
            for convention in self.conventions:
                for level in range(1, convention.get_num_levels() + 1):
                    opening_quote = convention.get_opening_quote_at_level(level)
                    closing_quote = convention.get_closing_quote_at_level(level)
                    opening_quotation_marks.add(opening_quote)
                    closing_quotation_marks.add(closing_quote)
                    all_quotation_marks.add(opening_quote)
                    all_quotation_marks.add(closing_quote)

            self.opening_quotation_mark_regex: Pattern = regex.compile(r"[" + "".join(opening_quotation_marks) + "]")
            self.closing_quotation_mark_regex: Pattern = regex.compile(r"[" + "".join(closing_quotation_marks) + "]")
            self.all_quotation_mark_regex: Pattern = regex.compile(r"[" + "".join(all_quotation_marks) + "]")
        else:
            self.opening_quotation_mark_regex = regex.compile(r"")
            self.closing_quotation_mark_regex = regex.compile(r"")
            self.all_quotation_mark_regex = regex.compile(r"")

    def _create_quotation_mark_pair_map(self) -> None:
        self.closing_marks_by_opening_mark: Dict[str, set[str]] = dict()
        self.opening_marks_by_closing_mark: Dict[str, set[str]] = dict()
        for convention in self.conventions:
            for level in range(1, convention.get_num_levels() + 1):
                opening_quote = convention.get_opening_quote_at_level(level)
                closing_quote = convention.get_closing_quote_at_level(level)
                if opening_quote not in self.closing_marks_by_opening_mark:
                    self.closing_marks_by_opening_mark[opening_quote] = set()
                self.closing_marks_by_opening_mark[opening_quote].add(closing_quote)
                if closing_quote not in self.opening_marks_by_closing_mark:
                    self.opening_marks_by_closing_mark[closing_quote] = set()
                self.opening_marks_by_closing_mark[closing_quote].add(opening_quote)

    def get_quote_convention_by_name(self, name: str) -> Union[QuoteConvention, None]:
        for convention in self.conventions:
            if convention.get_name() == name:
                return convention
        return None

    def get_possible_opening_marks(self) -> list[str]:
        return list(self.closing_marks_by_opening_mark.keys())

    def get_possible_closing_marks(self) -> list[str]:
        return list(self.opening_marks_by_closing_mark.keys())

    def is_valid_opening_quotation_mark(self, quotation_mark: str) -> bool:
        return quotation_mark in self.closing_marks_by_opening_mark

    def is_valid_closing_quotation_mark(self, quotation_mark: str) -> bool:
        for closing_mark_set in self.closing_marks_by_opening_mark.values():
            if quotation_mark in closing_mark_set:
                return True
        return False

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
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

    def get_opening_quotation_mark_regex(self) -> Pattern:
        return self.opening_quotation_mark_regex

    def get_closing_quotation_mark_regex(self) -> Pattern:
        return self.closing_quotation_mark_regex

    def get_quotation_mark_regex(self) -> Pattern:
        return self.all_quotation_mark_regex

    def filter_to_compatible_quote_conventions(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> "QuoteConventionSet":
        return QuoteConventionSet(
            [
                convention
                for convention in self.conventions
                if convention.is_compatible_with_observed_quotation_marks(
                    opening_quotation_marks, closing_quotation_marks
                )
            ]
        )

    def find_most_similar_convention(
        self, tabulated_quotation_marks: QuotationMarkTabulator
    ) -> Tuple[Union[QuoteConvention, None], float]:
        best_similarity: float = float("-inf")
        best_quote_convention: Union[QuoteConvention, None] = None
        for quote_convention in self.conventions:
            similarity = tabulated_quotation_marks.calculate_similarity(quote_convention)
            if similarity > best_similarity:
                best_similarity = similarity
                best_quote_convention = quote_convention

        return (best_quote_convention, best_similarity)

    def print_summary(self) -> None:
        print("Opening quotation marks must be one of the following: ", self.get_possible_opening_marks())
        print("Closing quotation marks must be one of the following: ", self.get_possible_closing_marks())
