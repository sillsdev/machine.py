from typing import List, Optional, Set, Union

from ..scripture.canon import book_id_to_number
from .paratext_project_file_handler import ParatextProjectFileHandler
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .usfm_parser import parse_usfm
from .usfm_versification_error_detector import UsfmVersificationError, UsfmVersificationErrorDetector


class ParatextProjectVersificationErrorDetectorBase:
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

    def get_usfm_versification_errors(
        self, handler: Optional[UsfmVersificationErrorDetector] = None, books: Optional[Set[int]] = None
    ) -> List[UsfmVersificationError]:
        handler = handler or UsfmVersificationErrorDetector(self._settings)
        for book_id in self._settings.get_all_scripture_book_ids():

            file_name = self._settings.get_book_file_name(book_id)

            if not self._paratext_project_file_handler.exists(file_name):
                continue

            if books is not None and not book_id_to_number(book_id) in books:
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
        return handler.errors
