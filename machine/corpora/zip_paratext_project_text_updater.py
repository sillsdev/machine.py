from io import BytesIO
from typing import BinaryIO, Optional
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .paratext_project_text_updater_base import ParatextProjectTextUpdaterBase
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser


class ZipParatextProjectTextUpdater(ParatextProjectTextUpdaterBase):
    def __init__(self, archive: ZipFile) -> None:
        super().__init__(ZipParatextProjectSettingsParser(archive))

        self._archive = archive

    def _exists(self, file_name: StrPath) -> bool:
        return file_name in self._archive.namelist()

    def _open(self, file_name: StrPath) -> Optional[BinaryIO]:
        if file_name in self._archive.namelist():
            return BytesIO(self._archive.read(file_name))
        return None
