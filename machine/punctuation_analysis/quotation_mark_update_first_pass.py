from typing import Dict, List, Set

from .chapter import Chapter
from .depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .quotation_mark_finder import QuotationMarkFinder
from .quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .quotation_mark_resolver import QuotationMarkResolver
from .quotation_mark_string_match import QuotationMarkStringMatch
from .quotation_mark_update_resolution_settings import QuotationMarkUpdateResolutionSettings
from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy
from .quote_convention import QuoteConvention
from .quote_convention_set import QuoteConventionSet
from .usfm_structure_extractor import UsfmStructureExtractor


# Determines the best strategy to take for each chapter
class QuotationMarkUpdateFirstPass(UsfmStructureExtractor):

    def __init__(self, old_quote_convention: QuoteConvention, new_quote_convention: QuoteConvention):
        super().__init__()
        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([old_quote_convention])
        )
        self._quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            QuotationMarkUpdateResolutionSettings(old_quote_convention)
        )
        self._will_fallback_mode_work: bool = self._check_whether_fallback_mode_will_work(
            old_quote_convention, new_quote_convention
        )

    def _check_whether_fallback_mode_will_work(
        self, old_quote_convention: QuoteConvention, new_quote_convention: QuoteConvention
    ) -> bool:
        new_opening_marks_by_old_marks: Dict[str, str] = {}
        new_closing_marks_by_old_marks: Dict[str, str] = {}
        for depth in range(1, min(old_quote_convention.num_levels, new_quote_convention.num_levels) + 1):
            old_opening_quotation_mark = old_quote_convention.get_opening_quotation_mark_at_depth(depth)
            new_opening_quotation_mark = new_quote_convention.get_opening_quotation_mark_at_depth(depth)
            if (
                old_opening_quotation_mark in new_opening_marks_by_old_marks
                and new_opening_marks_by_old_marks[old_opening_quotation_mark] != new_opening_quotation_mark
            ):
                return False
            new_opening_marks_by_old_marks[old_opening_quotation_mark] = new_opening_quotation_mark

            old_closing_quotation_mark = old_quote_convention.get_closing_quotation_mark_at_depth(depth)
            new_closing_quotation_mark = new_quote_convention.get_closing_quotation_mark_at_depth(depth)
            if (
                old_closing_quotation_mark in new_closing_marks_by_old_marks
                and new_closing_marks_by_old_marks[old_closing_quotation_mark] != new_closing_quotation_mark
            ):
                return False
            new_closing_marks_by_old_marks[old_closing_quotation_mark] = new_closing_quotation_mark

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
