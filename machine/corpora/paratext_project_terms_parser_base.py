import re
from abc import ABC, abstractmethod
from collections import defaultdict
from importlib.resources import open_binary
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree

from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_settings_parser_base import ParatextProjectSettingsParserBase

_PREDEFINED_TERMS_LIST_TYPES = ["Major", "All", "SilNt", "Pt6"]
_SUPPORTED_LANGUAGE_TERMS_LOCALIZATION_XMLS_PACKAGE = "machine.corpora"
_SUPPORTED_LANGUAGE_TERMS_LOCALIZATION_XMLS = {
    "en": "BiblicalTermsEn.xml",
    "es": "BiblicalTermsEs.xml",
    "fr": "BiblicalTermsFr.xml",
    "id": "BiblicalTermsId.xml",
    "pt": "BiblicalTermsPt.xml",
}
_CONTENT_IN_BRACKETS_REGEX = re.compile(r"^\[(.+?)\]$")
_NUMERICAL_INFORMATION_REGEX = re.compile(r"\s+\d+(\.\d+)*$")


class ParatextProjectTermsParserBase(ABC):
    def __init__(self, settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase]) -> None:
        self._settings: ParatextProjectSettings
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def parse(self, term_categories: Sequence[str], use_term_glosses: bool = True) -> List[Tuple[str, List[str]]]:
        biblical_terms_doc = None
        if self._settings.biblical_terms_list_type == "Project":
            if self._exists(self._settings.biblical_terms_file_name):
                with self._open(self._settings.biblical_terms_file_name) as stream:
                    biblical_terms_doc = ElementTree.parse(stream)
                    term_id_to_category_dict = _get_category_per_id(biblical_terms_doc)
        elif self._settings.biblical_terms_list_type in _PREDEFINED_TERMS_LIST_TYPES:
            with open_binary(
                _SUPPORTED_LANGUAGE_TERMS_LOCALIZATION_XMLS_PACKAGE, self._settings.biblical_terms_file_name
            ) as stream:
                biblical_terms_doc = ElementTree.parse(stream)
                term_id_to_category_dict = _get_category_per_id(biblical_terms_doc)
        else:
            term_id_to_category_dict = {}

        terms_glosses_doc: Optional[ElementTree.ElementTree] = None
        resource_name = None
        if self._settings.language_code is not None:
            resource_name = _SUPPORTED_LANGUAGE_TERMS_LOCALIZATION_XMLS.get(self._settings.language_code)
        if (
            self._settings.language_code is not None
            and self._settings.biblical_terms_list_type == "Major"
            and resource_name
        ):
            with open_binary(_SUPPORTED_LANGUAGE_TERMS_LOCALIZATION_XMLS_PACKAGE, resource_name) as stream:
                terms_glosses_doc = ElementTree.parse(stream)

        term_renderings_doc: Optional[ElementTree.ElementTree] = None
        if self._exists("TermRenderings.xml"):
            with self._open("TermRenderings.xml") as stream:
                term_renderings_doc = ElementTree.parse(stream)

        terms_renderings: Dict[str, List[str]] = defaultdict(list)
        if term_renderings_doc is not None:
            for term in term_renderings_doc.findall(".//TermRendering"):
                id = term.attrib["Id"]
                if _is_in_category(id, term_categories, term_id_to_category_dict):
                    id_ = id.replace("\n", "&#xA")
                    renderings = term.find("Renderings")
                    gloss = renderings.text if renderings is not None and renderings.text is not None else ""
                    glosses = _get_glosses(gloss)
                    terms_renderings[id_].extend(glosses)

        terms_glosses: Dict[str, List[str]] = defaultdict(list)
        if terms_glosses_doc is not None and use_term_glosses:
            for elem in terms_glosses_doc.findall(".//Localization"):
                id = elem.attrib["Id"]
                if _is_in_category(id, term_categories, term_id_to_category_dict):
                    id_ = id.replace("\n", "&#xA")
                    gloss = elem.attrib["Gloss"]
                    glosses = _get_glosses(gloss)
                    terms_glosses[id_].extend(glosses)
        if terms_glosses or terms_renderings:
            combined = {**terms_renderings, **{k: v for k, v in terms_glosses.items() if k not in terms_renderings}}
            return [(key, list(value)) for key, value in combined.items()]

        return []

    @abstractmethod
    def _exists(self, file_name: str) -> bool: ...

    @abstractmethod
    def _open(self, file_name: str) -> BinaryIO: ...


def _is_in_category(id: str, term_categories: Sequence[str], term_id_to_category_dict: Dict[str, str]) -> bool:
    category = term_id_to_category_dict.get(id)
    return not term_categories or (category is not None and category in term_categories)


def _get_glosses(gloss: str) -> List[str]:
    match = _CONTENT_IN_BRACKETS_REGEX.match(gloss)
    if match:
        gloss = match.group(0)
    gloss = gloss.replace("?", "")
    gloss = gloss.replace("*", "")
    gloss = gloss.replace("/", " ")
    gloss = gloss.strip()
    gloss = _strip_parens(gloss)
    gloss = _strip_parens(gloss, left="[", right="]")
    gloss = gloss.strip()
    for match in _NUMERICAL_INFORMATION_REGEX.finditer(gloss):
        gloss = gloss.replace(match.group(0), "")
    glosses = re.split(r"\|\|", gloss)
    glosses = [re.split(r"[,;]", g) for g in glosses]
    glosses = [item.strip() for sublist in glosses for item in sublist if item.strip()]
    return glosses


def _strip_parens(term_string: str, left: str = "(", right: str = ")") -> str:
    parens: int = 0
    end: int = -1
    for i in range(len(term_string) - 1, -1, -1):
        c = term_string[i]
        if c == right:
            if parens == 0:
                end = i + 1
            parens += 1
        elif c == left:
            if parens > 0:
                parens -= 1
                if parens == 0:
                    term_string = term_string[:i] + term_string[end:]
    return term_string


def _get_category_per_id(biblical_terms_doc: ElementTree.ElementTree) -> Dict[str, str]:
    term_id_to_category_dict: Dict[str, str] = {}

    for term in biblical_terms_doc.findall(".//Term"):
        term_id = term.attrib["Id"]
        if term_id not in term_id_to_category_dict:
            category = term.find("Category")
            term_id_to_category_dict[term_id] = (
                category.text if category is not None and category.text is not None else ""
            )

    return term_id_to_category_dict
