from typing import Dict, List, Set

from .analysis.chapter import Chapter
from .analysis.depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .analysis.quotation_mark_finder import QuotationMarkFinder
from .analysis.quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .analysis.quotation_mark_resolver import QuotationMarkResolver
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .analysis.quote_convention import QuoteConvention
from .analysis.quote_convention_set import QuoteConventionSet
from .analysis.usfm_structure_extractor import UsfmStructureExtractor
from .quotation_mark_update_resolution_settings import QuotationMarkUpdateResolutionSettings
from .quotation_mark_update_strategy import QuotationMarkUpdateStrategy


class QuotationMarkUpdateFirstPass(UsfmStructureExtractor):

    def __init__(self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention):
        super().__init__()
        self._source_quote_convention: QuoteConvention = source_quote_convention
        self._target_quote_convention: QuoteConvention = target_quote_convention
        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([source_quote_convention])
        )
        self._quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            QuotationMarkUpdateResolutionSettings(source_quote_convention, target_quote_convention)
        )
        self._will_fallback_mode_work: bool = self._check_whether_fallback_mode_will_work(
            source_quote_convention, target_quote_convention
        )

    def _check_whether_fallback_mode_will_work(
        self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention
    ) -> bool:
        target_marks_by_source_marks: Dict[str, Set[str]] = {}
        for level in range(1, source_quote_convention.get_num_levels() + 1):
            opening_quotation_mark = source_quote_convention.get_opening_quote_at_level(level)
            if opening_quotation_mark not in target_marks_by_source_marks:
                target_marks_by_source_marks[opening_quotation_mark] = set()
            if level <= target_quote_convention.get_num_levels():
                target_marks_by_source_marks[opening_quotation_mark].add(
                    target_quote_convention.get_closing_quote_at_level(level)
                )

        for source_mark in target_marks_by_source_marks:
            if len(target_marks_by_source_marks[source_mark]) > 1:
                return False
        return True

    def get_best_actions_by_chapter(self) -> List[QuotationMarkUpdateStrategy]:
        best_actions_by_chapter: List[QuotationMarkUpdateStrategy] = []

        for chapter in self.get_chapters():
            best_actions_by_chapter.append(self._find_best_action_for_chapter(chapter))

        return best_actions_by_chapter

    def _find_best_action_for_chapter(self, chapter: Chapter) -> QuotationMarkUpdateStrategy:
        quotation_mark_matches: List[QuotationMarkStringMatch] = (
            self._quotation_mark_finder.find_all_potential_quotation_marks_in_chapter(chapter)
        )

        self._quotation_mark_resolver.reset()

        # use list() to force evaluation of the generator
        list(self._quotation_mark_resolver.resolve_quotation_marks(quotation_mark_matches))

        return self._choose_best_action_based_on_observed_issues(self._quotation_mark_resolver.get_issues())

    def _choose_best_action_based_on_observed_issues(self, issues) -> QuotationMarkUpdateStrategy:
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
