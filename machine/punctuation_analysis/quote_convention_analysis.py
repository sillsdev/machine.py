from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention


class QuoteConventionAnalysis:

    def __init__(
        self,
        convention_scores: dict[QuoteConvention, float],
        tabulated_quotation_marks: QuotationMarkTabulator,
        analysis_weight: float = 1.0,
    ):
        self._convention_scores = convention_scores
        self._best_quote_convention = max(convention_scores.items(), key=lambda item: item[1])[0]
        self._best_quote_convention_score = convention_scores[self._best_quote_convention]
        self._tabulated_quotation_marks = tabulated_quotation_marks
        self._analysis_weight = analysis_weight

    def get_all_convention_scores(self) -> list[tuple[QuoteConvention, float]]:
        return list(self._convention_scores.items())

    @property
    def analysis_summary(self) -> str:
        return self._tabulated_quotation_marks.get_summary_message()

    @property
    def best_quote_convention(self) -> QuoteConvention:
        return self._best_quote_convention

    @property
    def best_quote_convention_score(self) -> float:
        return self._best_quote_convention_score

    @property
    def weight(self) -> float:
        return self._analysis_weight

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
