from re import Pattern
from typing import Union

import regex

from .quotation_mark_direction import QuotationMarkDirection
from .quotation_mark_metadata import QuotationMarkMetadata
from .quote_convention_set import QuoteConventionSet
from .text_segment import TextSegment
from .usfm_marker_type import UsfmMarkerType


class QuotationMarkStringMatch:

    # extra stuff in the regex to handle Western Cham
    letter_pattern: Pattern = regex.compile(r"[\p{L}\U0001E200-\U0001E28F]", regex.U)
    latin_letter_pattern: Pattern = regex.compile(r"^\p{script=Latin}$", regex.U)
    whitespace_pattern: Pattern = regex.compile(r"[\s~]", regex.U)
    punctuation_pattern: Pattern = regex.compile(r"[\.,;\?!\)\]\-—۔،؛]", regex.U)
    quote_introducer_pattern: Pattern = regex.compile(r"[:,]\\s*", regex.U)

    def __init__(self, text_segment: TextSegment, start_index: int, end_index: int):
        self.text_segment = text_segment
        self.start_index = start_index
        self.end_index = end_index

    def get_quotation_mark(self) -> str:
        return self.text_segment.get_text()[self.start_index : self.end_index]

    def is_valid_opening_quotation_mark(self, quote_convention_set: QuoteConventionSet) -> bool:
        return quote_convention_set.is_valid_opening_quotation_mark(self.get_quotation_mark())

    def is_valid_closing_quotation_mark(self, quote_convention_set: QuoteConventionSet) -> bool:
        return quote_convention_set.is_valid_closing_quotation_mark(self.get_quotation_mark())

    def does_quotation_mark_match(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self.get_quotation_mark()) is not None

    def does_next_character_match(self, regex_pattern: regex.Pattern) -> bool:
        return self.get_next_character() is not None and regex_pattern.search(self.get_next_character()) is not None

    def does_previous_character_match(self, regex_pattern: regex.Pattern) -> bool:
        return (
            self.get_previous_character() is not None
            and regex_pattern.search(self.get_previous_character()) is not None
        )

    def get_previous_character(self) -> Union[str, None]:
        if self.start_index == 0:
            previous_segment = self.text_segment.get_previous_segment()
            if previous_segment is not None and not self.text_segment.is_marker_in_preceding_context(
                UsfmMarkerType.ParagraphMarker
            ):
                return previous_segment.get_text()[-1]
            return None
        return self.text_segment.get_text()[self.start_index - 1]

    def get_next_character(self) -> Union[str, None]:
        if self.is_at_end_of_segment():
            next_segment = self.text_segment.get_next_segment()
            if next_segment is not None and not next_segment.is_marker_in_preceding_context(
                UsfmMarkerType.ParagraphMarker
            ):
                return next_segment.get_text()[0]
            return None
        return self.text_segment.get_text()[self.end_index]

    def does_leading_substring_match(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self.text_segment.substring_before(self.start_index)) is not None

    def does_trailing_substring_match(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self.text_segment.substring_after(self.end_index)) is not None

    # this assumes that the two matches occur in the same verse
    def precedes(self, other: "QuotationMarkStringMatch") -> bool:
        return self.text_segment.index_in_verse < other.text_segment.index_in_verse or (
            self.text_segment.index_in_verse == other.text_segment.index_in_verse
            and self.start_index < other.start_index
        )

    def get_text_segment(self) -> TextSegment:
        return self.text_segment

    def get_start_index(self) -> int:
        return self.start_index

    def get_end_index(self) -> int:
        return self.end_index

    def get_context(self) -> str:
        return self.text_segment.get_text()[
            max(self.start_index - 10, 0) : min(self.end_index + 10, len(self.text_segment.get_text()))
        ]

    def resolve(self, depth: int, direction: QuotationMarkDirection) -> QuotationMarkMetadata:
        return QuotationMarkMetadata(
            self.get_quotation_mark(), depth, direction, self.text_segment, self.start_index, self.end_index
        )

    def is_at_start_of_segment(self) -> bool:
        return self.start_index == 0

    def is_at_end_of_segment(self) -> bool:
        return self.end_index == self.text_segment.length()

    def has_leading_whitespace(self) -> bool:
        if self.get_previous_character() is None:
            return (
                self.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.ParagraphMarker)
                or self.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.EmbedMarker)
                or self.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.VerseMarker)
            )

        return self.does_previous_character_match(self.whitespace_pattern)

    def has_trailing_whitespace(self) -> bool:
        return self.does_next_character_match(self.whitespace_pattern)

    def has_leading_punctuation(self) -> bool:
        return self.does_next_character_match(self.punctuation_pattern)

    def has_trailing_punctuation(self) -> bool:
        return self.does_next_character_match(self.punctuation_pattern)

    def has_letter_in_leading_substring(self) -> bool:
        return self.does_leading_substring_match(self.letter_pattern)

    def has_letter_in_trailing_substring(self) -> bool:
        return self.does_trailing_substring_match(self.letter_pattern)

    def has_leading_latin_letter(self) -> bool:
        return self.does_previous_character_match(self.latin_letter_pattern)

    def has_trailing_latin_letter(self) -> bool:
        return self.does_next_character_match(self.latin_letter_pattern)

    def has_quote_introducer_in_leading_substring(self) -> bool:
        return self.does_leading_substring_match(self.quote_introducer_pattern)

    def has_leading_closing_quotation_mark(self, quote_convention_set: QuoteConventionSet) -> bool:
        return self.does_previous_character_match(quote_convention_set.get_opening_quotation_mark_regex())

    def has_trailing_closing_quotation_mark(self, quote_convention_set: QuoteConventionSet) -> bool:
        return self.does_next_character_match(quote_convention_set.get_closing_quotation_mark_regex())
