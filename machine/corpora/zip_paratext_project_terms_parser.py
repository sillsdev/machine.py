from io import BytesIO
from typing import BinaryIO, Optional
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_terms_parser_base import ParatextProjectTermsParserBase
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ZipParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, archive: ZipFile, settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(settings or ZipParatextProjectSettingsParser(archive).parse())

        self._archive = archive

    def _exists(self, file_name: StrPath) -> bool:
        return file_name in self._archive.namelist()

    def _open(self, file_name: StrPath) -> Optional[BinaryIO]:
        if file_name in self._archive.namelist():
            return BytesIO(self._archive.read(file_name))
        return None
