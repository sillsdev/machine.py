from zipfile import ZipFile

from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from .zip_paratext_project_file_handler import ZipParatextProjectFileHandler


class ZipParatextProjectSettingsParser(ParatextProjectSettingsParserBase):
    def __init__(self, archive: ZipFile) -> None:
        super().__init__(ZipParatextProjectFileHandler(archive))
