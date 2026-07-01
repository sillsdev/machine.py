from typing import Dict, Optional, Set, Union

from ..scripture.canon import book_id_to_number
from .paratext_project_file_handler import ParatextProjectFileHandler
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .usfm_parser import parse_usfm
from .usfm_versification_analyzer_handler import UsfmVersificationAnalysis, UsfmVersificationAnalyzerHandler


class UsfmVersificationAnalyzerBase:
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

    def analyze_usfm_versification(
        self,
        book_ids_and_chapters: Optional[Dict[str, Optional[Set[int]]]] = None,
        handler: Optional[UsfmVersificationAnalyzerHandler] = None,
    ) -> UsfmVersificationAnalysis:
        book_nums_and_chapters = (
            {book_id_to_number(book_id): chapters for book_id, chapters in book_ids_and_chapters.items()}
            if book_ids_and_chapters is not None
            else None
        )
        handler = handler or UsfmVersificationAnalyzerHandler(self._settings, book_nums_and_chapters)
        for book_id in self._settings.get_all_scripture_book_ids():

            file_name = self._settings.get_book_file_name(book_id)

            if not self._paratext_project_file_handler.exists(file_name):
                continue

            if book_nums_and_chapters is not None and book_id_to_number(book_id) not in book_nums_and_chapters:
                continue

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
        return handler.get_analysis()
