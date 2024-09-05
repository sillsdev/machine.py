from abc import ABC, abstractmethod
from typing import BinaryIO
from xml.etree import ElementTree

from ..scripture.verse_ref import Versification
from ..utils.string_utils import parse_integer
from .corpora_utils import get_encoding
from .paratext_project_settings import ParatextProjectSettings
from .usfm_stylesheet import UsfmStylesheet


class ParatextProjectSettingsParserBase(ABC):

    @abstractmethod
    def _exists(self, file_name: str) -> bool: ...

    @abstractmethod
    def _find(self, extension: str) -> str: ...

    @abstractmethod
    def _open(self, file_name: str) -> BinaryIO: ...

    @abstractmethod
    def _create_stylesheet(self, file_name: str) -> UsfmStylesheet: ...

    def parse(self) -> ParatextProjectSettings:
        settings_file_name = "Settings.xml"
        if not self._exists(settings_file_name):
            settings_file_name = self._find(".ssf")
        if not settings_file_name:
            raise ValueError("The project does not contain a settings file.")
        with self._open(settings_file_name) as stream:
            settings_tree = ElementTree.parse(stream)

        name = settings_tree.getroot().findtext("Name", "")
        full_name = settings_tree.getroot().findtext("FullName", "")
        encoding_str = settings_tree.getroot().findtext("Encoding", "65001")
        code_page = parse_integer(encoding_str)
        if code_page is None:
            raise NotImplementedError(
                f"The project uses a legacy encoding that requires TECKit, map file: {encoding_str}."
            )
        encoding = get_encoding(code_page)
        if encoding is None:
            raise RuntimeError(f"Code page {code_page} not supported.")

        versification_type = int(settings_tree.getroot().findtext("Versification", "4"))
        versification = Versification.get_builtin(versification_type)
        if self._exists("custom.vrs"):
            guid = settings_tree.getroot().findtext("Guid", "")
            versification_name = f"{versification.name}-{guid}"
            versification = Versification.load(
                self._open("custom.vrs"),
                versification,
                versification_name,
            )
        stylesheet_file_name = settings_tree.getroot().findtext("StyleSheet", "usfm.sty")
        if not self._exists(stylesheet_file_name) and stylesheet_file_name != "usfm_sb.sty":
            stylesheet_file_name = "usfm.sty"
        stylesheet = self._create_stylesheet(stylesheet_file_name)

        prefix = ""
        form = "41MAT"
        suffix = ".SFM"
        naming_elem = settings_tree.getroot().find("Naming")
        if naming_elem is not None:
            pre_part = naming_elem.get("PrePart")
            if pre_part:
                prefix = pre_part
            book_name_form = naming_elem.get("BookNameForm")
            if book_name_form:
                form = book_name_form
            post_part = naming_elem.get("PostPart")
            if post_part:
                suffix = post_part
        biblical_terms_list_setting = settings_tree.getroot().findtext("BiblicalTermsListSetting", "")
        if biblical_terms_list_setting is None:
            # Default to Major::BiblicalTerms.xml to mirror Paratext behavior
            biblical_terms_list_setting = "Major::BiblicalTerms.xml"
        parts = biblical_terms_list_setting.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"The BiblicalTermsListSetting element in Settings.xml in project {full_name}"
                f" is not in the expected format (e.g., Major::BiblicalTerms.xml) but is {biblical_terms_list_setting}."
            )
        language_code = None
        language_iso_code_setting = settings_tree.getroot().findtext("LanguageIsoCode", "")
        if language_iso_code_setting:
            language_iso_code_setting_parts = settings_tree.getroot().findtext("LanguageIsoCode", "").split(":")
            if language_iso_code_setting_parts:
                language_code = language_iso_code_setting_parts[0]

        return ParatextProjectSettings(
            name,
            full_name,
            encoding,
            versification,
            stylesheet,
            prefix,
            form,
            suffix,
            parts[0],
            parts[1],
            parts[2],
            language_code,
        )
