from typing import List

import regex

from .chapter import Chapter
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention_set import QuoteConventionSet
from .text_segment import TextSegment
from .verse import Verse


class QuotationMarkFinder:
    _QUOTATION_MARK_PATTERN = regex.compile(r"(\p{Quotation_Mark}|<<|>>|<|>)", regex.U)

    def __init__(self, quote_conventions: QuoteConventionSet):
        self._quote_conventions = quote_conventions

    def find_all_potential_quotation_marks_in_chapter(self, chapter: Chapter) -> List[QuotationMarkStringMatch]:
        quotation_matches: List[QuotationMarkStringMatch] = []
        for verse in chapter.verses:
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_verse(verse))
        return quotation_matches

    def find_all_potential_quotation_marks_in_verse(self, verse: Verse) -> List[QuotationMarkStringMatch]:
        return self.find_all_potential_quotation_marks_in_text_segments(verse.text_segments)

    def find_all_potential_quotation_marks_in_text_segments(
        self, text_segments: List[TextSegment]
    ) -> list[QuotationMarkStringMatch]:
        quotation_matches: List[QuotationMarkStringMatch] = []
        for text_segment in text_segments:
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_text_segment(text_segment))
        return quotation_matches

    def find_all_potential_quotation_marks_in_text_segment(
        self, text_segment: TextSegment
    ) -> List[QuotationMarkStringMatch]:
        quotation_matches: List[QuotationMarkStringMatch] = []
        for quotation_mark_match in self._QUOTATION_MARK_PATTERN.finditer(str(text_segment.text)):
            if self._quote_conventions.is_valid_opening_quotation_mark(
                quotation_mark_match.group()
            ) or self._quote_conventions.is_valid_closing_quotation_mark(quotation_mark_match.group()):
                quotation_matches.append(
                    QuotationMarkStringMatch(
                        text_segment,
                        quotation_mark_match.start(),
                        quotation_mark_match.end(),
                    )
                )
        return quotation_matches
