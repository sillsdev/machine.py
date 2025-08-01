from typing import Dict, List, Set

from .punctuation_analysis.chapter import Chapter
from .punctuation_analysis.depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .punctuation_analysis.quotation_mark_finder import QuotationMarkFinder
from .punctuation_analysis.quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .punctuation_analysis.quotation_mark_resolver import QuotationMarkResolver
from .punctuation_analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .punctuation_analysis.quote_convention import QuoteConvention
from .punctuation_analysis.quote_convention_set import QuoteConventionSet
from .punctuation_analysis.usfm_structure_extractor import UsfmStructureExtractor
from .quotation_mark_update_resolution_settings import QuotationMarkUpdateResolutionSettings
from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy


# Determines the best strategy to take for each chapter
class QuotationMarkUpdateFirstPass(UsfmStructureExtractor):

    def __init__(self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention):
        super().__init__()
        self._source_quote_convention: QuoteConvention = source_quote_convention
        self._target_quote_convention: QuoteConvention = target_quote_convention
        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([source_quote_convention])
        )
        self._quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            QuotationMarkUpdateResolutionSettings(source_quote_convention)
        )
        self._will_fallback_mode_work: bool = self._check_whether_fallback_mode_will_work(
            source_quote_convention, target_quote_convention
        )

    def _check_whether_fallback_mode_will_work(
        self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention
    ) -> bool:
        opening_target_marks_by_source_marks: Dict[str, str] = {}
        closing_target_marks_by_source_marks: Dict[str, str] = {}
        for depth in range(1, min(source_quote_convention.num_levels, target_quote_convention.num_levels) + 1):
            source_opening_quotation_mark = source_quote_convention.get_opening_quotation_mark_at_depth(depth)
            target_opening_quotation_mark = target_quote_convention.get_opening_quotation_mark_at_depth(depth)
            if (
                source_opening_quotation_mark in opening_target_marks_by_source_marks
                and opening_target_marks_by_source_marks[source_opening_quotation_mark] != target_opening_quotation_mark
            ):
                return False
            opening_target_marks_by_source_marks[source_opening_quotation_mark] = target_opening_quotation_mark

            source_closing_quotation_mark = source_quote_convention.get_closing_quotation_mark_at_depth(depth)
            target_closing_quotation_mark = target_quote_convention.get_closing_quotation_mark_at_depth(depth)
            if (
                source_closing_quotation_mark in closing_target_marks_by_source_marks
                and closing_target_marks_by_source_marks[source_closing_quotation_mark] != target_closing_quotation_mark
            ):
                return False
            closing_target_marks_by_source_marks[source_closing_quotation_mark] = target_closing_quotation_mark

        return True

    def find_best_chapter_strategies(self) -> List[QuotationMarkUpdateStrategy]:
        best_actions_by_chapter: List[QuotationMarkUpdateStrategy] = []

        for chapter in self.get_chapters():
            best_actions_by_chapter.append(self._find_best_strategy_for_chapter(chapter))

        return best_actions_by_chapter

    def _find_best_strategy_for_chapter(self, chapter: Chapter) -> QuotationMarkUpdateStrategy:
        quotation_mark_matches: List[QuotationMarkStringMatch] = (
            self._quotation_mark_finder.find_all_potential_quotation_marks_in_chapter(chapter)
        )

        self._quotation_mark_resolver.reset()

        # Use list() to force evaluation of the generator
        list(self._quotation_mark_resolver.resolve_quotation_marks(quotation_mark_matches))

        return self._choose_best_strategy_based_on_observed_issues(self._quotation_mark_resolver.get_issues())

    def _choose_best_strategy_based_on_observed_issues(
        self, issues: Set[QuotationMarkResolutionIssue]
    ) -> QuotationMarkUpdateStrategy:
        if QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK in issues:
            return QuotationMarkUpdateStrategy.SKIP

        if (
            QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK in issues
            or QuotationMarkResolutionIssue.TOO_DEEP_NESTING in issues
        ):
            if self._will_fallback_mode_work:
                return QuotationMarkUpdateStrategy.APPLY_FALLBACK
            return QuotationMarkUpdateStrategy.SKIP

        return QuotationMarkUpdateStrategy.APPLY_FULL
