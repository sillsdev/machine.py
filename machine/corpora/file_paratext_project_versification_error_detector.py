from typing import Optional

from ..utils.typeshed import StrPath
from .file_paratext_project_file_handler import FileParatextProjectFileHandler
from .file_paratext_project_settings_parser import FileParatextProjectSettingsParser
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_versification_error_detector_base import ParatextProjectVersificationErrorDetectorBase


class FileParatextProjectVersificationErrorDetector(ParatextProjectVersificationErrorDetectorBase):
    def __init__(self, project_dir: StrPath, parent_settings: Optional[ParatextProjectSettings] = None) -> None:
        super().__init__(
            FileParatextProjectFileHandler(project_dir),
            FileParatextProjectSettingsParser(project_dir, parent_settings).parse(),
        )
