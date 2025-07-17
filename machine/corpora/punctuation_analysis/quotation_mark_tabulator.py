from collections import Counter, defaultdict
from typing import List

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quote_convention import QuoteConvention


class QuotationMarkCounts:
    def __init__(self):
        self._quotation_mark_counter: Counter[str] = Counter()
        self._total_count = 0

    def count_quotation_mark(self, quotation_mark: str) -> None:
        self._quotation_mark_counter.update([quotation_mark])
        self._total_count += 1

    def find_best_quotation_mark_proportion(self) -> tuple[str, int, int]:
        return self._quotation_mark_counter.most_common(1)[0] + (self._total_count,)

    def calculate_num_differences(self, expected_quotation_mark: str) -> int:
        return self._total_count - self._quotation_mark_counter[expected_quotation_mark]

    def get_observed_count(self) -> int:
        return self._total_count


class QuotationMarkTabulator:

    def __init__(self):
        self._quotation_counts_by_depth_and_direction: dict[tuple[int, QuotationMarkDirection], QuotationMarkCounts] = (
            defaultdict(QuotationMarkCounts)
        )

    def tabulate(self, quotation_marks: list[QuotationMarkMetadata]) -> None:
        for quotation_mark in quotation_marks:
            self._count_quotation_mark(quotation_mark)

    def _count_quotation_mark(self, quotation_mark: QuotationMarkMetadata) -> None:
        key = (quotation_mark.depth, quotation_mark.direction)
        self._quotation_counts_by_depth_and_direction[key].count_quotation_mark(quotation_mark.quotation_mark)

    def _depth_and_direction_observed(self, depth: int, direction: QuotationMarkDirection) -> bool:
        return (depth, direction) in self._quotation_counts_by_depth_and_direction

    def _find_most_common_quotation_mark_with_depth_and_direction(
        self, depth: int, direction: QuotationMarkDirection
    ) -> tuple[str, int, int]:
        return self._quotation_counts_by_depth_and_direction[(depth, direction)].find_best_quotation_mark_proportion()

    def calculate_similarity(self, quote_convention: QuoteConvention) -> float:
        weighted_difference = 0
        total_weight = 0
        for depth, direction in self._quotation_counts_by_depth_and_direction:
            expected_quotation_mark: str = quote_convention.get_expected_quotation_mark(depth, direction)

            # Give higher weight to shallower depths, since deeper marks are more likely to be mistakes
            weighted_difference += self._quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].calculate_num_differences(expected_quotation_mark) * 2 ** (-depth)
            total_weight += self._quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].get_observed_count() * 2 ** (-depth)

        if total_weight == 0:
            return 0
        return 1 - (weighted_difference / total_weight)

    def get_summary_message(self) -> str:
        message_lines: List[str] = []
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
                message_lines.append(
                    (
                        "The most common level %i quotation marks are "
                        + "%s (%i of %i opening marks) and %s (%i of %i closing marks)"
                    )
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
        return "\n".join(message_lines)
