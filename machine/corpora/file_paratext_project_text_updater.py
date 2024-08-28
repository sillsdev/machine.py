from pathlib import Path
from typing import BinaryIO

from machine.corpora.file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from machine.corpora.paratext_project_text_updater_base import ParatextProjectTextUpdaterBase

from ..utils.typeshed import StrPath


class FileParatextProjectTextUpdater(ParatextProjectTextUpdaterBase):
    def __init__(self, project_dir: StrPath) -> None:
        super().__init__(FileParatextProjectSettingsParser(project_dir))

        self._project_dir = project_dir

    def exists(self, file_name: StrPath) -> bool:
        return (Path(self._project_dir) / file_name).exists()

    def open(self, file_name: StrPath) -> BinaryIO:
        return open(Path(self._project_dir) / file_name, mode="rb")
