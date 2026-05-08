from typing import Dict, Optional

from machine.corpora import MemoryParatextProjectFileHandler, ParatextProjectSettings, ParatextProjectTermsParserBase

from .default_paratext_project_settings import DefaultParatextProjectSettings


class MemoryParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, files: Dict[str, str], settings: Optional[ParatextProjectSettings]) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings or DefaultParatextProjectSettings())
