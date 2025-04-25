from typing import Dict, List, Set

from .analysis.depth_based_quotation_mark_resolver import DepthBasedQuotationMarkResolver
from .analysis.quotation_mark_finder import QuotationMarkFinder
from .analysis.quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .analysis.quotation_mark_resolver import QuotationMarkResolver
from .analysis.quotation_mark_string_match import QuotationMarkStringMatch
from .analysis.quote_convention import QuoteConvention
from .analysis.quote_convention_set import QuoteConventionSet
from .analysis.usfm_structure_extractor import UsfmStructureExtractor
from .quotation_denormalization_action import QuotationDenormalizationAction
from .quotation_denormalization_resolution_settings import QuotationDenormalizationResolutionSettings


class QuotationDenormalizationFirstPass(UsfmStructureExtractor):

    def __init__(self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention):
        super().__init__()
        self._quotation_mark_finder: QuotationMarkFinder = QuotationMarkFinder(
            QuoteConventionSet([source_quote_convention.normalize()])
        )
        self._quotation_mark_resolver: QuotationMarkResolver = DepthBasedQuotationMarkResolver(
            QuotationDenormalizationResolutionSettings(source_quote_convention, target_quote_convention)
        )
        self._will_basic_denormalization_work: bool = self._check_whether_basic_denormalization_will_work(
            source_quote_convention, target_quote_convention
        )

    def _check_whether_basic_denormalization_will_work(
        self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention
    ) -> bool:
        normalized_source_quote_convention: QuoteConvention = source_quote_convention.normalize()
        target_marks_by_normalized_source_marks: Dict[str, Set[str]] = {}
        for level in range(1, normalized_source_quote_convention.get_num_levels() + 1):
            normalized_opening_quotation_mark = normalized_source_quote_convention.get_opening_quote_at_level(level)
            if normalized_opening_quotation_mark not in target_marks_by_normalized_source_marks:
                target_marks_by_normalized_source_marks[normalized_opening_quotation_mark] = set()
            target_marks_by_normalized_source_marks[normalized_opening_quotation_mark].add(
                target_quote_convention.get_closing_quote_at_level(level)
            )

        for normalized_source_mark in target_marks_by_normalized_source_marks:
            if len(target_marks_by_normalized_source_marks[normalized_source_mark]) > 1:
                return False
        return True

    def get_best_actions_by_chapter(self, usfm_text: str) -> List[QuotationDenormalizationAction]:
        best_actions_by_chapter: List[QuotationDenormalizationAction] = []

        for chapter in self.get_chapters():
            best_actions_by_chapter.append(self._find_best_action_for_chapter(chapter))

        return best_actions_by_chapter

    def _find_best_action_for_chapter(self, chapter) -> QuotationDenormalizationAction:
        quotation_mark_matches: List[QuotationMarkStringMatch] = (
            self._quotation_mark_finder.find_all_potential_quotation_marks_in_chapter(chapter)
        )

        self._quotation_mark_resolver.reset()
        list(self._quotation_mark_resolver.resolve_quotation_marks(quotation_mark_matches))

        return self._choose_best_action_based_on_observed_issues(self._quotation_mark_resolver.get_issues())

    def _choose_best_action_based_on_observed_issues(self, issues) -> QuotationDenormalizationAction:
        print(issues)
        if (
            QuotationMarkResolutionIssue.AMBIGUOUS_QUOTATION_MARK in issues
            or QuotationMarkResolutionIssue.UNEXPECTED_QUOTATION_MARK in issues
        ):
            return QuotationDenormalizationAction.SKIP

        if (
            QuotationMarkResolutionIssue.UNPAIRED_QUOTATION_MARK in issues
            or QuotationMarkResolutionIssue.TOO_DEEP_NESTING in issues
            or QuotationMarkResolutionIssue.INCOMPATIBLE_QUOTATION_MARK in issues
        ):
            if self._will_basic_denormalization_work:
                return QuotationDenormalizationAction.APPLY_BASIC
            return QuotationDenormalizationAction.SKIP

        return QuotationDenormalizationAction.APPLY_FULL
