from abc import ABC, abstractmethod
from collections import defaultdict
from typing import BinaryIO, Dict, List, Optional, Union

from ..corpora.paratext_project_settings import ParatextProjectSettings
from ..corpora.paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from ..corpora.usfm_parser import parse_usfm
from ..scripture.canon import book_id_to_number, get_scripture_books
from ..utils.typeshed import StrPath
from .quotation_mark_tabulator import QuotationMarkTabulator
from .quote_convention import QuoteConvention
from .quote_convention_analysis import QuoteConventionAnalysis
from .quote_convention_detector import QuoteConventionDetector


class WeightedAverageQuoteConventionAnalysisBuilder:
    def __init__(self) -> None:
        self._total_weight: float = 0
        self._convention_votes: Dict[str, float] = defaultdict(float)
        self._quote_conventions_by_name: Dict[str, QuoteConvention] = {}
        self._total_tabulated_quotation_marks = QuotationMarkTabulator()

    def record_book_results(
        self,
        quote_convention_analysis: QuoteConventionAnalysis,
        tabulated_quotation_marks: QuotationMarkTabulator,
    ) -> None:
        if quote_convention_analysis.best_quote_convention is None or quote_convention_analysis.weight == 0:
            return

        self._total_tabulated_quotation_marks.tabulate_from(tabulated_quotation_marks)

        self._total_weight += quote_convention_analysis.weight
        for convention, score in quote_convention_analysis.get_all_convention_scores():
            if convention.name not in self._quote_conventions_by_name:
                self._quote_conventions_by_name[convention.name] = convention
            self._convention_votes[convention.name] += score * quote_convention_analysis.weight

    def to_quote_convention_analysis(self) -> QuoteConventionAnalysis:
        quote_convention_analysis_builder = QuoteConventionAnalysis.Builder(self._total_tabulated_quotation_marks)

        for convention_name, total_score in self._convention_votes.items():
            if total_score > 0:
                quote_convention_analysis_builder.record_convention_score(
                    self._quote_conventions_by_name[convention_name], total_score / self._total_weight
                )

        return quote_convention_analysis_builder.build()


class ParatextProjectQuoteConventionDetector(ABC):
    def __init__(self, settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase]) -> None:
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def get_quote_convention_analysis(
        self, include_chapters: Optional[Dict[int, List[int]]] = None
    ) -> QuoteConventionAnalysis:

        weighted_average_quote_convention_analysis_builder = WeightedAverageQuoteConventionAnalysisBuilder()

        for book_id in get_scripture_books():
            if include_chapters is not None and book_id_to_number(book_id) not in include_chapters:
                continue
            file_name: str = self._settings.get_book_file_name(book_id)
            if not self._exists(file_name):
                continue

            handler = QuoteConventionDetector()

            with self._open(file_name) as sfm_file:
                usfm: str = sfm_file.read().decode(self._settings.encoding)
            try:
                parse_usfm(usfm, handler, self._settings.stylesheet, self._settings.versification)
            except Exception as e:
                error_message = (
                    f"An error occurred while parsing the usfm for '{file_name}'"
                    f"{f' in project {self._settings.name}' if self._settings.name else ''}"
                    f". Error: '{e}'"
                )
                raise RuntimeError(error_message) from e

            quote_convention_analysis, tabulated_quotation_marks = (
                handler.detect_quote_convention_and_get_tabulated_quotation_marks(include_chapters)
            )
            weighted_average_quote_convention_analysis_builder.record_book_results(
                quote_convention_analysis, tabulated_quotation_marks
            )

        return weighted_average_quote_convention_analysis_builder.to_quote_convention_analysis()

    @abstractmethod
    def _exists(self, file_name: StrPath) -> bool: ...

    @abstractmethod
    def _open(self, file_name: StrPath) -> BinaryIO: ...
