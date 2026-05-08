from typing import Dict

from machine.corpora import MemoryParatextProjectFileHandler, ParatextProjectSettings
from machine.punctuation_analysis import ParatextProjectQuoteConventionDetector

from .default_paratext_project_settings import DefaultParatextProjectSettings


class MemoryParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, settings: ParatextProjectSettings, files: Dict[str, str]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
