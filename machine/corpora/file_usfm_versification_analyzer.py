from typing import Optional

from ..utils.typeshed import StrPath
from .file_paratext_project_file_handler import FileParatextProjectFileHandler
from .file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from .paratext_project_settings import ParatextProjectSettings
from .usfm_versification_analyzer_base import UsfmVersificationAnalyzerBase


class FileUsfmVersificationAnalyzer(UsfmVersificationAnalyzerBase):
    def __init__(self, project_dir: StrPath, parent_settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(
            FileParatextProjectFileHandler(project_dir),
            FileParatextProjectSettingsParser(project_dir, parent_settings).parse(),
        )
