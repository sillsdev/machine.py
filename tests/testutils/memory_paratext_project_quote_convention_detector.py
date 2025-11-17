from io import BytesIO
from typing import BinaryIO, Dict

from machine.corpora import ParatextProjectSettings
from machine.punctuation_analysis import ParatextProjectQuoteConventionDetector

from .memory_paratext_project_file_handler import MemoryParatextProjectFileHandler


class MemoryParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, settings: ParatextProjectSettings, files: Dict[str, str]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
