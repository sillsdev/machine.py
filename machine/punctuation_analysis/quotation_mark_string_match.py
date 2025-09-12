from re import Pattern
from typing import Optional

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quote_convention_set import QuoteConventionSet
from .text_segment import TextSegment
from .usfm_marker_type import UsfmMarkerType


class QuotationMarkStringMatch:

    # Extra stuff in the regex to handle Western Cham
    _LETTER_PATTERN: Pattern = regex.compile(r"[\p{L}\U0001E200-\U0001E28F]", regex.U)
    _LATIN_LETTER_PATTERN: Pattern = regex.compile(r"^\p{script=Latin}$", regex.U)
    _WHITESPACE_PATTERN: Pattern = regex.compile(r"[\s~]", regex.U)
    _PUNCTUATION_PATTERN: Pattern = regex.compile(r"[\.,;\?!\)\]\-—۔،؛]", regex.U)
    _QUOTE_INTRODUCER_PATTERN: Pattern = regex.compile(r"[:,]\s*$", regex.U)

    def __init__(self, text_segment: TextSegment, start_index: int, end_index: int):
        self._text_segment = text_segment
        self._start_index = start_index
        self._end_index = end_index

    def __eq__(self, value):
        if not isinstance(value, QuotationMarkStringMatch):
            return False
        return (
            self._text_segment == value._text_segment
            and self._start_index == value._start_index
            and self._end_index == value._end_index
        )

    @property
    def quotation_mark(self) -> str:
        return str(self._text_segment.text[self._start_index : self._end_index])

    def is_valid_opening_quotation_mark(self, quote_conventions: QuoteConventionSet) -> bool:
        return quote_conventions.is_valid_opening_quotation_mark(self.quotation_mark)

    def is_valid_closing_quotation_mark(self, quote_conventions: QuoteConventionSet) -> bool:
        return quote_conventions.is_valid_closing_quotation_mark(self.quotation_mark)

    def quotation_mark_matches(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self.quotation_mark) is not None

    def next_character_matches(self, regex_pattern: regex.Pattern) -> bool:
        return self.next_character is not None and regex_pattern.search(self.next_character) is not None

    def previous_character_matches(self, regex_pattern: regex.Pattern) -> bool:
        return self.previous_character is not None and regex_pattern.search(self.previous_character) is not None

    @property
    def previous_character(self) -> Optional[str]:
        if self.is_at_start_of_segment():
            previous_segment = self._text_segment.previous_segment
            if previous_segment is not None and not self._text_segment.marker_is_in_preceding_context(
                UsfmMarkerType.PARAGRAPH
            ):
                return str(previous_segment.text[-1])
            return None
        return str(self._text_segment.text[self._start_index - 1])

    @property
    def next_character(self) -> Optional[str]:
        if self.is_at_end_of_segment():
            next_segment = self._text_segment.next_segment
            if next_segment is not None and not next_segment.marker_is_in_preceding_context(UsfmMarkerType.PARAGRAPH):
                return str(next_segment.text[0])
            return None
        return str(self._text_segment.text[self._end_index])

    def leading_substring_matches(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self._text_segment.substring_before(self._start_index)) is not None

    def trailing_substring_matches(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self._text_segment.substring_after(self._end_index)) is not None

    # This assumes that the two matches occur in the same verse
    def precedes(self, other: "QuotationMarkStringMatch") -> bool:
        return self._text_segment.index_in_verse < other._text_segment.index_in_verse or (
            self._text_segment.index_in_verse == other._text_segment.index_in_verse
            and self._start_index < other._start_index
        )

    @property
    def text_segment(self) -> TextSegment:
        return self._text_segment

    @property
    def start_index(self) -> int:
        return self._start_index

    @property
    def end_index(self) -> int:
        return self._end_index

    # Not used, but a useful method for debugging
    @property
    def context(self) -> str:
        return str(
            self._text_segment.text[
                max(self._start_index - 10, 0) : min(self._end_index + 10, len(self._text_segment.text))
            ]
        )

    def resolve(self, depth: int, direction: QuotationMarkDirection) -> QuotationMarkMetadata:
        return QuotationMarkMetadata(
            self.quotation_mark, depth, direction, self._text_segment, self._start_index, self._end_index
        )

    def is_at_start_of_segment(self) -> bool:
        return self._start_index == 0

    def is_at_end_of_segment(self) -> bool:
        return self._end_index == self._text_segment.length

    def has_leading_whitespace(self) -> bool:
        if self.previous_character is None:
            return (
                self._text_segment.marker_is_in_preceding_context(UsfmMarkerType.PARAGRAPH)
                or self._text_segment.marker_is_in_preceding_context(UsfmMarkerType.EMBED)
                or self._text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE)
            )

        return self.previous_character_matches(self._WHITESPACE_PATTERN)

    def has_trailing_whitespace(self) -> bool:
        return self.next_character_matches(self._WHITESPACE_PATTERN)

    def has_leading_punctuation(self) -> bool:
        return self.previous_character_matches(self._PUNCTUATION_PATTERN)

    def has_trailing_punctuation(self) -> bool:
        return self.next_character_matches(self._PUNCTUATION_PATTERN)

    def has_letter_in_leading_substring(self) -> bool:
        return self.leading_substring_matches(self._LETTER_PATTERN)

    def has_letter_in_trailing_substring(self) -> bool:
        return self.trailing_substring_matches(self._LETTER_PATTERN)

    def has_leading_latin_letter(self) -> bool:
        return self.previous_character_matches(self._LATIN_LETTER_PATTERN)

    def has_trailing_latin_letter(self) -> bool:
        return self.next_character_matches(self._LATIN_LETTER_PATTERN)

    def has_quote_introducer_in_leading_substring(self) -> bool:
        return self.leading_substring_matches(self._QUOTE_INTRODUCER_PATTERN)
