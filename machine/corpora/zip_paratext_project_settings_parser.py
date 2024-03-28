from io import BytesIO
from typing import BinaryIO, Optional
from zipfile import ZipFile

from .zip_paratext_project_settings_parser_base import ZipParatextProjectSettingsParserBase


class ZipParatextProjectSettingsParser(ZipParatextProjectSettingsParserBase):
    def __init__(self, archive: ZipFile) -> None:
        self._archive = archive

    def exists(self, file_name: str) -> bool:
        return file_name in self._archive.namelist()

    def find(self, extension: str) -> Optional[str]:
        for entry in self._archive.namelist():
            if entry.endswith(extension):
                return entry
        return None

    def open(self, file_name: str) -> Optional[BinaryIO]:
        if file_name in self._archive.namelist():
            return BytesIO(self._archive.read(file_name))
        return None
