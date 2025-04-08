from typing import Dict

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quote_convention import QuoteConvention


class QuotationMarkCounts:
    def __init__(self):
        self.string_counts: Dict[str, int] = dict()
        self.total_count = 0

    def count_quotation_mark(self, quotation_mark: str) -> None:
        if quotation_mark not in self.string_counts:
            self.string_counts[quotation_mark] = 0
        self.string_counts[quotation_mark] += 1
        self.total_count += 1

    def get_best_proportion(self) -> tuple[str, int, int]:
        best_str = max(self.string_counts, key=lambda x: self.string_counts[x])
        return (best_str, self.string_counts[best_str], self.total_count)

    def calculate_num_differences(self, expected_quotation_mark: str) -> int:
        if expected_quotation_mark not in self.string_counts:
            return self.total_count
        return self.total_count - self.string_counts[expected_quotation_mark]

    def get_observed_count(self) -> int:
        return self.total_count


class QuotationMarkTabulator:

    def __init__(self):
        self.quotation_counts_by_depth_and_direction: dict[tuple[int, QuotationMarkDirection], QuotationMarkCounts] = (
            dict()
        )

    def tabulate(self, quotation_marks: list[QuotationMarkMetadata]) -> None:
        for quotation_mark in quotation_marks:
            self._count_quotation_mark(quotation_mark)

    def _count_quotation_mark(self, quote: QuotationMarkMetadata) -> None:
        key = (quote.get_depth(), quote.get_direction())
        quotation_mark = quote.get_quotation_mark()
        if key not in self.quotation_counts_by_depth_and_direction:
            self.quotation_counts_by_depth_and_direction[key] = QuotationMarkCounts()
        self.quotation_counts_by_depth_and_direction[key].count_quotation_mark(quotation_mark)

    def _has_depth_and_direction_been_observed(self, depth: int, direction: QuotationMarkDirection) -> bool:
        return (depth, direction) in self.quotation_counts_by_depth_and_direction

    def _get_most_common_quote_by_depth_and_direction(
        self, depth: int, direction: QuotationMarkDirection
    ) -> tuple[str, int, int]:
        return self.quotation_counts_by_depth_and_direction[(depth, direction)].get_best_proportion()

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
            if self._has_depth_and_direction_been_observed(
                depth, QuotationMarkDirection.Opening
            ) and self._has_depth_and_direction_been_observed(depth, QuotationMarkDirection.Closing):
                (opening_quotation_mark, observed_opening_count, total_opening_count) = (
                    self._get_most_common_quote_by_depth_and_direction(depth, QuotationMarkDirection.Opening)
                )
                (closing_quotation_mark, observed_closing_count, total_closing_count) = (
                    self._get_most_common_quote_by_depth_and_direction(depth, QuotationMarkDirection.Closing)
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
