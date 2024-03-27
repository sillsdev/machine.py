from abc import ABC

from ..scripture.verse_ref import Versification
from .usfm_stylesheet import UsfmStylesheet


class ParatextProjectSettings(ABC):
    def __init__(
        self,
        name: str,
        full_name: str,
        encoding: str,
        versification: Versification,
        stylesheet: UsfmStylesheet,
        file_name_prefix: str,
        file_name_form: str,
        file_name_suffix: str,
        biblical_terms_list_type: str,
        biblical_terms_project_name: str,
        biblical_terms_file_name: str,
    ) -> None:
        self._name = name
        self._full_name = full_name
        self._encoding = encoding
        self._versification = versification
        self._stylesheet = stylesheet
        self._file_name_prefix = file_name_prefix
        self._file_name_form = file_name_form
        self._file_name_suffix = file_name_suffix
        self._biblical_terms_list_type = biblical_terms_list_type
        self._biblical_terms_project_name = biblical_terms_project_name
        self._biblical_terms_file_name = biblical_terms_file_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def full_name(self) -> str:
        return self._full_name

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def versification(self) -> Versification:
        return self._versification

    @property
    def stylesheet(self) -> UsfmStylesheet:
        return self._stylesheet

    @property
    def file_name_prefix(self) -> str:
        return self._file_name_prefix

    @property
    def file_name_form(self) -> str:
        return self._file_name_form

    @property
    def file_name_suffix(self) -> str:
        return self._file_name_suffix

    @property
    def biblical_terms_list_type(self) -> str:
        return self._biblical_terms_list_type

    @property
    def biblical_terms_project_name(self) -> str:
        return self._biblical_terms_project_name

    @property
    def biblical_terms_file_name(self) -> str:
        return self._biblical_terms_file_name
