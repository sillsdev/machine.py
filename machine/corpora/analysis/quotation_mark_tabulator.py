from typing import Dict

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quote_convention import QuoteConvention


class QuotationMarkCounts:
    def __init__(self):
        self._string_counts: Dict[str, int] = dict()
        self._total_count = 0

    def count_quotation_mark(self, quotation_mark: str) -> None:
        if quotation_mark not in self._string_counts:
            self._string_counts[quotation_mark] = 0
        self._string_counts[quotation_mark] += 1
        self._total_count += 1

    def find_best_quotation_mark_proportion(self) -> tuple[str, int, int]:
        best_str = max(self._string_counts, key=lambda x: self._string_counts[x])
        return (best_str, self._string_counts[best_str], self._total_count)

    def calculate_num_differences(self, expected_quotation_mark: str) -> int:
        if expected_quotation_mark not in self._string_counts:
            return self._total_count
        return self._total_count - self._string_counts[expected_quotation_mark]

    def get_observed_count(self) -> int:
        return self._total_count


class QuotationMarkTabulator:

    def __init__(self):
        self.quotation_counts_by_depth_and_direction: dict[tuple[int, QuotationMarkDirection], QuotationMarkCounts] = (
            dict()
        )

    def tabulate(self, quotation_marks: list[QuotationMarkMetadata]) -> None:
        for quotation_mark in quotation_marks:
            self._count_quotation_mark(quotation_mark)

    def _count_quotation_mark(self, quote: QuotationMarkMetadata) -> None:
        key = (quote.depth, quote.direction)
        quotation_mark = quote.quotation_mark
        if key not in self.quotation_counts_by_depth_and_direction:
            self.quotation_counts_by_depth_and_direction[key] = QuotationMarkCounts()
        self.quotation_counts_by_depth_and_direction[key].count_quotation_mark(quotation_mark)

    def _depth_and_direction_observed(self, depth: int, direction: QuotationMarkDirection) -> bool:
        return (depth, direction) in self.quotation_counts_by_depth_and_direction

    def _find_most_common_quotation_mark_with_depth_and_direction(
        self, depth: int, direction: QuotationMarkDirection
    ) -> tuple[str, int, int]:
        return self.quotation_counts_by_depth_and_direction[(depth, direction)].find_best_quotation_mark_proportion()

    def calculate_similarity(self, quote_convention: QuoteConvention) -> float:
        num_differences = 0
        num_total_quotation_marks = 0
        for depth, direction in self.quotation_counts_by_depth_and_direction:
            expected_quotation_mark: str = quote_convention.get_expected_quotation_mark(depth, direction)

            # give higher weight to shallower depths, since deeper marks are more likely to be mistakes
            num_differences += self.quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].calculate_num_differences(expected_quotation_mark) * 2 ** (-depth)
            num_total_quotation_marks += self.quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].get_observed_count() * 2 ** (-depth)

        if num_total_quotation_marks == 0:
            return 0
        return 1 - (num_differences / num_total_quotation_marks)

    def print_summary(self) -> None:
        for depth in range(1, 5):
            if self._depth_and_direction_observed(
                depth, QuotationMarkDirection.OPENING
            ) and self._depth_and_direction_observed(depth, QuotationMarkDirection.CLOSING):
                (opening_quotation_mark, observed_opening_count, total_opening_count) = (
                    self._find_most_common_quotation_mark_with_depth_and_direction(
                        depth, QuotationMarkDirection.OPENING
                    )
                )
                (closing_quotation_mark, observed_closing_count, total_closing_count) = (
                    self._find_most_common_quotation_mark_with_depth_and_direction(
                        depth, QuotationMarkDirection.CLOSING
                    )
                )
                print(
                    "The most common level %i quotes are %s (%i of %i opening quotes) and %s (%i of %i closing quotes)"
                    % (
                        depth,
                        opening_quotation_mark,
                        observed_opening_count,
                        total_opening_count,
                        closing_quotation_mark,
                        observed_closing_count,
                        total_closing_count,
                    )
                )
