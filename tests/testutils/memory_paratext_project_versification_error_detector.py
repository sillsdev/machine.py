from typing import Dict, Optional

from machine.corpora import ParatextProjectSettings, ParatextProjectVersificationErrorDetectorBase

from .memory_paratext_project_file_handler import DefaultParatextProjectSettings, MemoryParatextProjectFileHandler


class MemoryParatextProjectVersificationErrorDetector(ParatextProjectVersificationErrorDetectorBase):
    def __init__(self, settings: Optional[ParatextProjectSettings], files: Dict[str, str]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
