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
    def __init__(self, name: str, levels: list[SingleLevelQuoteConvention]):
        self._name = name
        self.levels = levels

    def __eq__(self, value):
        if not isinstance(value, QuoteConvention):
            return False
        if self._name != value._name:
            return False
        if len(self.levels) != len(value.levels):
            return False
        for level, other_level in zip(self.levels, value.levels):
            if level.opening_quotation_mark != other_level.opening_quotation_mark:
                return False
            if level.closing_quotation_mark != other_level.closing_quotation_mark:
                return False
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def get_opening_quotation_mark_at_level(self, level: int) -> str:
        return self.levels[level - 1].opening_quotation_mark

    def get_closing_quotation_mark_at_level(self, level: int) -> str:
        return self.levels[level - 1].closing_quotation_mark

    def get_expected_quotation_mark(self, depth: int, direction: QuotationMarkDirection) -> str:
        if depth > self.num_levels or depth < 1:
            return ""
        return (
            self.get_opening_quotation_mark_at_level(depth)
            if direction is QuotationMarkDirection.OPENING
            else self.get_closing_quotation_mark_at_level(depth)
        )

    def _includes_opening_quotation_mark(self, opening_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.opening_quotation_mark == opening_quotation_mark:
                return True
        return False

    def _includes_closing_quotation_mark(self, closing_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.closing_quotation_mark == closing_quotation_mark:
                return True
        return False

    def get_possible_depths(self, quotation_mark: str, direction: QuotationMarkDirection) -> Set[int]:
        depths: Set[int] = set()
        for depth, level in enumerate(self.levels, start=1):
            if direction is QuotationMarkDirection.OPENING and level.opening_quotation_mark == quotation_mark:
                depths.add(depth)
            elif direction is QuotationMarkDirection.CLOSING and level.closing_quotation_mark == quotation_mark:
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

        # we require the first-level quotation marks to have been observed
        if (
            self.get_opening_quotation_mark_at_level(1) not in opening_quotation_marks
            or self.get_closing_quotation_mark_at_level(1) not in closing_quotation_marks
        ):
            return False
        return True

    def normalize(self) -> "QuoteConvention":
        return QuoteConvention(self.name + "_normalized", [level.normalize() for level in self.levels])

    def __str__(self) -> str:
        summary = self.name + "\n"
        for level, convention in enumerate(self.levels):
            ordinal_name = self._get_ordinal_name(level + 1)
            summary += "%s%s-level quote%s\n" % (
                convention.opening_quotation_mark,
                ordinal_name,
                convention.closing_quotation_mark,
            )
        return summary

    def _get_ordinal_name(self, level) -> str:
        if level == 1:
            return "First"
        if level == 2:
            return "Second"
        if level == 3:
            return "Third"
        if level == 4:
            return "Fourth"
        return str(level) + "th"
