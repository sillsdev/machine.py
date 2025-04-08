from typing import Union

from .chapter import Chapter
from .preliminary_quotation_analyzer import PreliminaryQuotationAnalyzer
from .quotation_mark_finder import QuotationMarkFinder
from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_resolver import QuotationMarkResolver
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention
from .quote_convention_set import QuoteConventionSet
from .standard_quote_conventions import standard_quote_conventions
from .usfm_structure_extractor import UsfmStructureExtractor


class QuoteConventionAnalysis:
    def __init__(self, best_quote_convention: QuoteConvention, best_quote_convention_score: float):
        self.best_quote_convention = best_quote_convention
        self.best_quote_convention_score = best_quote_convention_score

    def get_best_quote_convention(self) -> QuoteConvention:
        return self.best_quote_convention

    def get_best_quote_convention_similarity_score(self) -> float:
        return self.best_quote_convention_score * 100


class QuoteConventionDetector(UsfmStructureExtractor):

    def __init__(self):
        super().__init__()
        self.quotation_mark_tabulator = QuotationMarkTabulator()

    def _count_quotation_marks_in_chapters(self, chapters: list[Chapter]) -> None:
        possible_quote_conventions: QuoteConventionSet = PreliminaryQuotationAnalyzer(
            standard_quote_conventions
        ).narrow_down_possible_quote_conventions(chapters)

        for chapter in chapters:
            self._count_quotation_marks_in_chapter(chapter, possible_quote_conventions)

    def _count_quotation_marks_in_chapter(
        self, chapter: Chapter, possible_quote_conventions: QuoteConventionSet
    ) -> None:
        quotation_mark_matches: list[QuotationMarkStringMatch] = QuotationMarkFinder(
            possible_quote_conventions
        ).find_all_potential_quotation_marks_in_chapter(chapter)

        resolved_quotation_marks: list[QuotationMarkMetadata] = list(
            QuotationMarkResolver(possible_quote_conventions).resolve_quotation_marks(quotation_mark_matches)
        )

        self.quotation_mark_tabulator.tabulate(resolved_quotation_marks)

    def detect_quotation_convention(self, print_summary: bool) -> Union[QuoteConventionAnalysis, None]:
        self._count_quotation_marks_in_chapters(self.get_chapters())

        (best_quote_convention, score) = standard_quote_conventions.find_most_similar_convention(
            self.quotation_mark_tabulator
        )

        if print_summary:
            self.quotation_mark_tabulator.print_summary()

        if score > 0 and best_quote_convention is not None:
            return QuoteConventionAnalysis(best_quote_convention, score)
        return None
