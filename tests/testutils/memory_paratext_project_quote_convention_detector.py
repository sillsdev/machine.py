from typing import Dict, Optional

from machine.corpora import ParatextProjectSettings
from machine.punctuation_analysis import ParatextProjectQuoteConventionDetector

from .memory_paratext_project_file_handler import DefaultParatextProjectSettings, MemoryParatextProjectFileHandler


class MemoryParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, settings: ParatextProjectSettings, files: Optional[Dict[str, str]]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
