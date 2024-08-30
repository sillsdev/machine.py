from io import BytesIO
from typing import BinaryIO, Dict

from ..corpora.paratext_project_settings import ParatextProjectSettings
from .paratext_project_terms_parser_base import ParatextProjectTermsParserBase


class MemoryParatextProjectTermsParser(ParatextProjectTermsParserBase):
    def __init__(self, settings: ParatextProjectSettings, files: Dict[str, str]) -> None:
        super().__init__(settings)

        self.files = files

    def exists(self, file_name: str) -> bool:
        return file_name in self.files

    def open(self, file_name: str) -> BinaryIO:
        return BytesIO(self.files[file_name].encode("utf-8"))
