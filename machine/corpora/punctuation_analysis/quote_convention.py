from dataclasses import dataclass
from typing import Dict, Set

from .quotation_mark_direction import QuotationMarkDirection

_QUOTATION_MARK_NORMALIZATION_MAP: Dict[str, str] = {
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u300a": '"',
    "\u300b": '"',
    "\u300c": '"',
    "\u300d": '"',
}


@dataclass(frozen=True)
class SingleLevelQuoteConvention:
    opening_quotation_mark: str
    closing_quotation_mark: str

    def normalize(self) -> "SingleLevelQuoteConvention":
        normalized_opening_quotation_mark = (
            _QUOTATION_MARK_NORMALIZATION_MAP[self.opening_quotation_mark]
            if self.opening_quotation_mark in _QUOTATION_MARK_NORMALIZATION_MAP
            else self.opening_quotation_mark
        )
        normalized_closing_quotation_mark = (
            _QUOTATION_MARK_NORMALIZATION_MAP[self.closing_quotation_mark]
            if self.closing_quotation_mark in _QUOTATION_MARK_NORMALIZATION_MAP
            else self.closing_quotation_mark
        )
        return SingleLevelQuoteConvention(normalized_opening_quotation_mark, normalized_closing_quotation_mark)


class QuoteConvention:
    def __init__(self, name: str, level_conventions: list[SingleLevelQuoteConvention]):
        self._name = name
        self.level_conventions = level_conventions

    def __eq__(self, value):
        if not isinstance(value, QuoteConvention):
            return False
        if self._name != value._name:
            return False
        if len(self.level_conventions) != len(value.level_conventions):
            return False
        for level_convention, other_level_convention in zip(self.level_conventions, value.level_conventions):
            if level_convention.opening_quotation_mark != other_level_convention.opening_quotation_mark:
                return False
            if level_convention.closing_quotation_mark != other_level_convention.closing_quotation_mark:
                return False
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_levels(self) -> int:
        return len(self.level_conventions)

    def get_opening_quotation_mark_at_depth(self, depth: int) -> str:
        return self.level_conventions[depth - 1].opening_quotation_mark

    def get_closing_quotation_mark_at_depth(self, depth: int) -> str:
        return self.level_conventions[depth - 1].closing_quotation_mark

    def get_expected_quotation_mark(self, depth: int, direction: QuotationMarkDirection) -> str:
        if depth > self.num_levels or depth < 1:
            return ""
        return (
            self.get_opening_quotation_mark_at_depth(depth)
            if direction is QuotationMarkDirection.OPENING
            else self.get_closing_quotation_mark_at_depth(depth)
        )

    def _includes_opening_quotation_mark(self, opening_quotation_mark: str) -> bool:
        for level_convention in self.level_conventions:
            if level_convention.opening_quotation_mark == opening_quotation_mark:
                return True
        return False

    def _includes_closing_quotation_mark(self, closing_quotation_mark: str) -> bool:
        for level_convention in self.level_conventions:
            if level_convention.closing_quotation_mark == closing_quotation_mark:
                return True
        return False

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        depths: Set[int] = set()
        for depth, level_convention in enumerate(self.level_conventions, start=1):
            if (
                direction is QuotationMarkDirection.OPENING
                and level_convention.opening_quotation_mark == quotation_mark
            ):
                depths.add(depth)
            elif (
                direction is QuotationMarkDirection.CLOSING
                and level_convention.closing_quotation_mark == quotation_mark
            ):
                depths.add(depth)
        return depths

    def is_compatible_with_observed_quotation_marks(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> bool:
        for opening_quotation_mark in opening_quotation_marks:
            if not self._includes_opening_quotation_mark(opening_quotation_mark):
                return False
        for closing_quotation_mark in closing_quotation_marks:
            if not self._includes_closing_quotation_mark(closing_quotation_mark):
                return False

        # We require the first-level quotation marks to have been observed
        if (
            self.get_opening_quotation_mark_at_depth(1) not in opening_quotation_marks
            or self.get_closing_quotation_mark_at_depth(1) not in closing_quotation_marks
        ):
            return False
        return True

    def normalize(self) -> "QuoteConvention":
        return QuoteConvention(
            self.name + "_normalized", [level_convention.normalize() for level_convention in self.level_conventions]
        )

    def __str__(self) -> str:
        summary = self.name + "\n"
        for depth, level_convention in enumerate(self.level_conventions):
            ordinal_name = self._get_ordinal_name(depth + 1)
            summary += "%s%s-level quote%s\n" % (
                level_convention.opening_quotation_mark,
                ordinal_name,
                level_convention.closing_quotation_mark,
            )
        return summary

    def _get_ordinal_name(self, depth) -> str:
        if depth == 1:
            return "First"
        if depth == 2:
            return "Second"
        if depth == 3:
            return "Third"
        if depth == 4:
            return "Fourth"
        return str(depth) + "th"
