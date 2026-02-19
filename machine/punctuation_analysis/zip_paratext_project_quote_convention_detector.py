from typing import Optional
from zipfile import ZipFile

from ..corpora.paratext_project_settings import ParatextProjectSettings
from ..corpora.zip_paratext_project_file_handler import ZipParatextProjectFileHandler
from ..corpora.zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser
from .paratext_project_quote_convention_detector import ParatextProjectQuoteConventionDetector


class ZipParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, archive: ZipFile, parent_settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(
            ZipParatextProjectFileHandler(archive), ZipParatextProjectSettingsParser(archive, parent_settings).parse()
        )
