from io import BytesIO
from typing import BinaryIO, Dict

from machine.corpora import ParatextProjectSettings

from machine.corpora.paratext_project_quote_convention_detector import ParatextProjectQuoteConventionDetector


class MemoryParatextProjectQuoteConventionDetector(ParatextProjectQuoteConventionDetector):
    def __init__(self, settings: ParatextProjectSettings, files: Dict[str, str]) -> None:
        super().__init__(settings)

        self.files = files

    def _exists(self, file_name: str) -> bool:
        return file_name in self.files

    def _open(self, file_name: str) -> BinaryIO:
        return BytesIO(self.files[file_name].encode("utf-8"))
