from typing import Dict, Optional

from machine.corpora import ParatextProjectSettings, ParatextProjectTermsParserBase

from .memory_paratext_project_file_handler import DefaultParatextProjectSettings, MemoryParatextProjectFileHandler


class MemoryParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, files: Dict[str, str], settings: Optional[ParatextProjectSettings]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
