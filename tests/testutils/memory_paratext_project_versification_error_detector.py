from typing import Dict, Optional

from machine.corpora import ParatextProjectSettings

from machine.corpora.paratext_project_versification_error_detector import ParatextProjectVersificationErrorDetector
from .memory_paratext_project_file_handler import DefaultParatextProjectSettings, MemoryParatextProjectFileHandler


class MemoryParatextProjectVersificationErrorDetector(ParatextProjectVersificationErrorDetector):
    def __init__(self, settings: Optional[ParatextProjectSettings], files: Dict[str, str]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
