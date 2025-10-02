from dataclasses import dataclass
from typing import Dict, List, Optional

from .chapter import Chapter
from .depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .preliminary_quotation_mark_analyzer import PreliminaryQuotationMarkAnalyzer
from .quotation_mark_finder import QuotationMarkFinder
from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention
from .quote_convention_detection_resolution_settings import QuoteConventionDetectionResolutionSettings
from .quote_convention_set import QuoteConventionSet
from .standard_quote_conventions import STANDARD_QUOTE_CONVENTIONS
from .usfm_structure_extractor import UsfmStructureExtractor


@dataclass(frozen=True)
class QuoteConventionAnalysis:
    best_quote_convention: QuoteConvention
    best_quote_convention_score: float
    analysis_summary: str


class QuoteConventionDetector(UsfmStructureExtractor):

    def __init__(self):
        super().__init__()
        self._quotation_mark_tabulator = QuotationMarkTabulator()

    def _count_quotation_marks_in_chapters(self, chapters: list[Chapter]) -> None:
        possible_quote_conventions: QuoteConventionSet = PreliminaryQuotationMarkAnalyzer(
            STANDARD_QUOTE_CONVENTIONS
        ).narrow_down_possible_quote_conventions(chapters)

        for chapter in chapters:
            self._count_quotation_marks_in_chapter(chapter, possible_quote_conventions)

    def _count_quotation_marks_in_chapter(
        self, chapter: Chapter, possible_quote_conventions: QuoteConventionSet
    ) -> None:
        quotation_mark_matches: List[QuotationMarkStringMatch] = QuotationMarkFinder(
            possible_quote_conventions
        ).find_all_potential_quotation_marks_in_chapter(chapter)

        resolved_quotation_marks: List[QuotationMarkMetadata] = list(
            DepthBasedQuotationMarkResolver(
                QuoteConventionDetectionResolutionSettings(possible_quote_conventions)
            ).resolve_quotation_marks(quotation_mark_matches)
        )

        self._quotation_mark_tabulator.tabulate(resolved_quotation_marks)

    def detect_quote_convention(
        self, include_chapters: Optional[Dict[int, List[int]]] = None
    ) -> Optional[QuoteConventionAnalysis]:
        self._count_quotation_marks_in_chapters(self.get_chapters(include_chapters))

        (best_quote_convention, score) = STANDARD_QUOTE_CONVENTIONS.find_most_similar_convention(
            self._quotation_mark_tabulator
        )

        if score > 0 and best_quote_convention is not None:
            return QuoteConventionAnalysis(
                best_quote_convention, score, self._quotation_mark_tabulator.get_summary_message()
            )
        return None
