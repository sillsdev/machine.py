from typing import Dict, Optional

from machine.corpora import (
    MemoryParatextProjectFileHandler,
    ParatextProjectSettings,
    ParatextProjectVersificationErrorDetectorBase,
)

from .default_paratext_project_settings import DefaultParatextProjectSettings


class MemoryParatextProjectVersificationErrorDetector(ParatextProjectVersificationErrorDetectorBase):
    def __init__(self, settings: Optional[ParatextProjectSettings], files: Dict[str, str]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
