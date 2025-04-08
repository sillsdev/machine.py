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

    def find_all_potential_quotation_marks_in_chapter(self, chapter: Chapter) -> list[QuotationMarkStringMatch]:
        quotation_matches: list[QuotationMarkStringMatch] = []
        for verse in chapter.get_verses():
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_verse(verse))
        return quotation_matches

    def find_all_potential_quotation_marks_in_verse(self, verse: Verse) -> list[QuotationMarkStringMatch]:
        quotation_matches: list[QuotationMarkStringMatch] = []
        for text_segment in verse.get_text_segments():
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_text_segment(text_segment))
        return quotation_matches

    def find_all_potential_quotation_marks_in_text_segment(
        self, text_segment: TextSegment
    ) -> list[QuotationMarkStringMatch]:
        quotation_matches: list[QuotationMarkStringMatch] = []
        for quote_match in self.quote_pattern.finditer(text_segment.get_text()):
            if self.quote_convention_set.is_valid_opening_quotation_mark(
                quote_match.group()
            ) or self.quote_convention_set.is_valid_closing_quotation_mark(quote_match.group()):
                quotation_matches.append(QuotationMarkStringMatch(text_segment, quote_match.start(), quote_match.end()))
        return quotation_matches
