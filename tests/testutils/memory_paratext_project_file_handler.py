from io import BytesIO
from typing import BinaryIO, Dict, Optional

from machine.corpora import ParatextProjectFileHandler, ParatextProjectSettings, UsfmStylesheet
from machine.scripture import ORIGINAL_VERSIFICATION, Versification


class MemoryParatextProjectFileHandler(ParatextProjectFileHandler):
    def __init__(self, files: Dict[str, str]) -> None:

        self.files = files

    def exists(self, file_name: str) -> bool:
        return file_name in self.files

    def open(self, file_name: str) -> BinaryIO:
        return BytesIO(self.files[file_name].encode("utf-8"))

    def find(self, extension):
        raise NotImplementedError

    def create_stylesheet(self, file_name):
        raise NotImplementedError


class DefaultParatextProjectSettings(ParatextProjectSettings):
    def __init__(
        self,
        name: str = "Test",
        full_name: str = "TestProject",
        encoding: Optional[str] = None,
        versification: Optional[Versification] = None,
        stylesheet: Optional[UsfmStylesheet] = None,
        file_name_prefix: str = "",
        file_name_form: str = "41MAT",
        file_name_suffix: str = "Test.SFM",
        biblical_terms_list_type: str = "Project",
        biblical_terms_project_name: str = "Test",
        biblical_terms_file_name: str = "ProjectBiblicalTerms.xml",
        language_code: str = "en",
    ):

        super().__init__(
            name,
            full_name,
            encoding if encoding is not None else "utf-8",
            versification if versification is not None else ORIGINAL_VERSIFICATION,
            stylesheet if stylesheet is not None else UsfmStylesheet("usfm.sty"),
            file_name_prefix,
            file_name_form,
            file_name_suffix,
            biblical_terms_list_type,
            biblical_terms_project_name,
            biblical_terms_file_name,
            language_code,
        )
