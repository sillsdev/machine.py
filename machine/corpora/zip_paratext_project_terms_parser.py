from typing import Optional
from zipfile import ZipFile

from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_terms_parser_base import ParatextProjectTermsParserBase
from .zip_paratext_project_file_handler import ZipParatextProjectFileHandler
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ZipParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, archive: ZipFile, settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(
            ZipParatextProjectFileHandler(archive), settings or ZipParatextProjectSettingsParser(archive).parse()
        )
