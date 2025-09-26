from pathlib import Path
from typing import BinaryIO

from ..corpora.file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from ..utils.typeshed import StrPath
from .paratext_project_quote_convention_detector import ParatextProjectQuoteConventionDetector


class FileParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, project_dir: StrPath) -> None:
        super().__init__(FileParatextProjectSettingsParser(project_dir))

        self._project_dir = project_dir

    def _exists(self, file_name: str) -> bool:
        return (Path(self._project_dir) / file_name).exists()

    def _open(self, file_name: StrPath) -> BinaryIO:
        return open(Path(self._project_dir) / file_name, mode="rb")
