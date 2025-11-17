from typing import Dict, List, Optional, Union

from ..corpora.paratext_project_file_handler import ParatextProjectFileHandler
from ..corpora.paratext_project_settings import ParatextProjectSettings
from ..corpora.paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from ..corpora.usfm_parser import parse_usfm
from ..scripture.canon import book_id_to_number, get_scripture_books
from .quote_convention_analysis import QuoteConventionAnalysis
from .quote_convention_detector import QuoteConventionDetector


class ParatextProjectQuoteConventionDetector:
    def __init__(
        self,
        paratext_project_file_handler: ParatextProjectFileHandler,
        settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase],
    ) -> None:
        self._paratext_project_file_handler = paratext_project_file_handler
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def get_quote_convention_analysis(
        self, include_chapters: Optional[Dict[int, List[int]]] = None
    ) -> QuoteConventionAnalysis:

        book_quote_convention_analyses: List[QuoteConventionAnalysis] = []

        for book_id in get_scripture_books():
            if include_chapters is not None and book_id_to_number(book_id) not in include_chapters:
                continue
            file_name: str = self._settings.get_book_file_name(book_id)
            if not self._paratext_project_file_handler.exists(file_name):
                continue

            handler = QuoteConventionDetector()

            with self._paratext_project_file_handler.open(file_name) as sfm_file:
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

            quote_convention_analysis = handler.detect_quote_convention(include_chapters)
            book_quote_convention_analyses.append(quote_convention_analysis)

        return QuoteConventionAnalysis.combine_with_weighted_average(book_quote_convention_analyses)
