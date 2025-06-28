from typing import List

import regex

from .chapter import Chapter
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quote_convention_set import QuoteConventionSet
from .text_segment import TextSegment
from .verse import Verse


class QuotationMarkFinder:
    quote_pattern = regex.compile(r"(\p{Quotation_Mark}|<<|>>|<|>)", regex.U)

    def __init__(self, quote_convention_set: QuoteConventionSet):
        self.quote_convention_set = quote_convention_set

    def find_all_potential_quotation_marks_in_chapter(self, chapter: Chapter) -> List[QuotationMarkStringMatch]:
        quotation_matches: List[QuotationMarkStringMatch] = []
        for verse in chapter.get_verses():
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_verse(verse))
        return quotation_matches

    def find_all_potential_quotation_marks_in_verse(self, verse: Verse) -> List[QuotationMarkStringMatch]:
        return self.find_all_potential_quotation_marks_in_text_segments(verse.get_text_segments())

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
        for quote_match in self.quote_pattern.finditer(text_segment.get_text()):
            if self.quote_convention_set.is_valid_opening_quotation_mark(
                quote_match.group()
            ) or self.quote_convention_set.is_valid_closing_quotation_mark(quote_match.group()):
                quotation_matches.append(QuotationMarkStringMatch(text_segment, quote_match.start(), quote_match.end()))
        return quotation_matches
