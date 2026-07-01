from typing import Dict, Optional

from machine.corpora import ParatextProjectSettings, UsfmVersificationAnalyzerBase

from .memory_paratext_project_file_handler import DefaultParatextProjectSettings, MemoryParatextProjectFileHandler


class MemoryUsfmVersificationAnalyzer(UsfmVersificationAnalyzerBase):
    def __init__(self, settings: Optional[ParatextProjectSettings], files: Optional[Dict[str, str]]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
