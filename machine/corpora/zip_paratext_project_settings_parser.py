from io import BytesIO
from typing import Any, BinaryIO, Union
from zipfile import ZipFile

from .zip_paratext_project_settings_parser_base import ZipParatextProjectSettingsParserBase


class ZipParatextProjectSettingsParser(ZipParatextProjectSettingsParserBase):
    def __init__(self, archive: ZipFile) -> None:
        self._archive = archive

    def __enter__(self) -> "ZipParatextProjectSettingsParser":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None: ...

    def exists(self, file_name: str) -> bool:
        return file_name in self._archive.namelist()

    def find(self, extension: str) -> Union[str, None]:
        for entry in self._archive.namelist():
            if entry.endswith(extension):
                return entry
        return None

    def open(self, file_name: str) -> Union[BinaryIO, None]:
        if file_name in self._archive.namelist():
            return BytesIO(self._archive.read(file_name))
        return None
