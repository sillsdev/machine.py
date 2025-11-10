from collections import defaultdict
from typing import Dict, List, Optional

from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention


class QuoteConventionAnalysis:

    def __init__(
        self,
        convention_scores: dict[QuoteConvention, float],
        tabulated_quotation_marks: QuotationMarkTabulator,
        analysis_weight: float = 1.0,  # weight is used for combining scores for multiple books
    ):
        self._convention_scores = convention_scores
        if len(convention_scores) > 0:
            (self._best_quote_convention, self._best_quote_convention_score) = max(
                convention_scores.items(), key=lambda item: item[1]
            )
        else:
            self._best_quote_convention_score = 0
            self._best_quote_convention = None

        self._tabulated_quotation_marks = tabulated_quotation_marks
        self._analysis_weight = analysis_weight

    @property
    def analysis_summary(self) -> str:
        return self._tabulated_quotation_marks.get_summary_message()

    @property
    def best_quote_convention(self) -> Optional[QuoteConvention]:
        return self._best_quote_convention

    @property
    def best_quote_convention_score(self) -> float:
        return self._best_quote_convention_score

    class Builder:
        def __init__(self, tabulated_quotation_marks: QuotationMarkTabulator):
            self._convention_scores: dict[QuoteConvention, float] = {}
            self._tabulated_quotation_marks = tabulated_quotation_marks

        def record_convention_score(self, quote_convention: QuoteConvention, score: float) -> None:
            self._convention_scores[quote_convention] = score

        def build(self) -> "QuoteConventionAnalysis":
            return QuoteConventionAnalysis(
                self._convention_scores,
                self._tabulated_quotation_marks,
                self._tabulated_quotation_marks.get_total_quotation_mark_count(),
            )

    @staticmethod
    def combine_with_weighted_average(
        quote_convention_analyses: List["QuoteConventionAnalysis"],
    ) -> "QuoteConventionAnalysis":
        total_weight: float = 0
        convention_votes: Dict[str, float] = defaultdict(float)
        quote_conventions_by_name: Dict[str, QuoteConvention] = {}
        total_tabulated_quotation_marks = QuotationMarkTabulator()
        for quote_convention_analysis in quote_convention_analyses:
            total_tabulated_quotation_marks.tabulate_from(quote_convention_analysis._tabulated_quotation_marks)
            total_weight += quote_convention_analysis._analysis_weight
            for convention, score in quote_convention_analysis._convention_scores.items():
                if convention.name not in quote_conventions_by_name:
                    quote_conventions_by_name[convention.name] = convention
                convention_votes[convention.name] += score * quote_convention_analysis._analysis_weight

        quote_convention_analysis_builder = QuoteConventionAnalysis.Builder(total_tabulated_quotation_marks)

        for convention_name, total_score in convention_votes.items():
            if total_score > 0:
                quote_convention_analysis_builder.record_convention_score(
                    quote_conventions_by_name[convention_name], total_score / total_weight
                )

        return quote_convention_analysis_builder.build()
