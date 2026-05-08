from typing import Optional

from machine.corpora import ParatextProjectSettings, UsfmStylesheet
from machine.scripture import ORIGINAL_VERSIFICATION, Versification


class DefaultParatextProjectSettings(ParatextProjectSettings):
    def __init__(
        self,
        guid: str = "id",
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
        translation_type: str = "Standard",
        parent_guid: Optional[str] = None,
        parent_name: Optional[str] = None,
    ):

        super().__init__(
            guid,
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
            translation_type,
            parent_guid,
            parent_name,
        )
