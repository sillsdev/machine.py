from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Union

from ..corpora.paratext_project_settings import ParatextProjectSettings
from ..corpora.paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from ..corpora.usfm_parser import parse_usfm
from ..utils.typeshed import StrPath
from .quote_convention_detector import QuoteConventionAnalysis, QuoteConventionDetector


class ParatextProjectQuoteConventionDetector(ABC):
    def __init__(self, settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase]) -> None:
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def get_quote_convention_analysis(
        self, handler: Optional[QuoteConventionDetector] = None
    ) -> Optional[QuoteConventionAnalysis]:
        handler = QuoteConventionDetector() if handler is None else handler
        for file_name in self._settings.get_all_scripture_book_file_names():
            if not self._exists(file_name):
                continue
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
        return handler.detect_quote_convention()

    @abstractmethod
    def _exists(self, file_name: StrPath) -> bool: ...

    @abstractmethod
    def _open(self, file_name: StrPath) -> BinaryIO: ...
