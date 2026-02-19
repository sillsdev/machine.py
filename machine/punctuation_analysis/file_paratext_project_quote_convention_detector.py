from pathlib import Path
from typing import BinaryIO, Optional

from ..corpora.file_paratext_project_file_handler import FileParatextProjectFileHandler
from ..corpora.file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from ..corpora.paratext_project_settings import ParatextProjectSettings
from ..utils.typeshed import StrPath
from .paratext_project_quote_convention_detector import ParatextProjectQuoteConventionDetector


class FileParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, project_dir: StrPath, parent_settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(
            FileParatextProjectFileHandler(project_dir),
            FileParatextProjectSettingsParser(project_dir, parent_settings).parse(),
        )

        self._project_dir = project_dir

    def _exists(self, file_name: str) -> bool:
        return (Path(self._project_dir) / file_name).exists()

    def _open(self, file_name: StrPath) -> BinaryIO:
        return open(Path(self._project_dir) / file_name, mode="rb")
