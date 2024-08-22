import re
from typing import Dict, List, Optional
from xml.etree import ElementTree
from zipfile import ZipFile

from .corpora_utils import get_entry
from .dictionary_text_corpus import DictionaryTextCorpus
from .memory_text import MemoryText
from .text_row import TextRow
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser

_PREDEFINED_TERMS_LIST_TYPES = ["Major", "All", "SilNt", "Pt6"]


class ParatextBackupTermsCorpus(DictionaryTextCorpus):
    def __init__(self, filename: str, term_categories: List[str]) -> None:
        with ZipFile(filename, "r") as archive:
            terms_file_entry = get_entry(archive, "TermRenderings.xml")
            if terms_file_entry is None:
                return
            settings_parser = ZipParatextProjectSettingsParser(archive)
            settings = settings_parser.parse()

            with archive.open(terms_file_entry) as key_terms_file:
                term_renderings_tree = ElementTree.parse(key_terms_file)

            biblical_terms_file_entry = get_entry(archive, settings.biblical_terms_file_name)
            if settings.biblical_terms_list_type == "Project":
                if biblical_terms_file_entry is not None:
                    with archive.open(biblical_terms_file_entry) as key_terms_file:
                        biblical_terms_tree = ElementTree.parse(key_terms_file)
                        term_id_to_category_dict = _get_category_per_id(biblical_terms_tree)
                else:
                    with open("BiblicalTerms.xml", "rb") as key_terms_file:
                        biblical_terms_tree = ElementTree.parse(key_terms_file)
                        term_id_to_category_dict = _get_category_per_id(biblical_terms_tree)
            elif settings.biblical_terms_list_type in _PREDEFINED_TERMS_LIST_TYPES:
                with open(settings.biblical_terms_file_name, "rb") as key_terms_file:
                    biblical_terms_tree = ElementTree.parse(key_terms_file)
                    term_id_to_category_dict = _get_category_per_id(biblical_terms_tree)
            else:
                term_id_to_category_dict = {}

            terms_elements = term_renderings_tree.iter(".//TermRendering")
            text_id = (
                f"{settings.biblical_terms_list_type}:"
                f"{settings.biblical_terms_project_name}:"
                f"{settings.biblical_terms_file_name}"
            )
            rows: List[TextRow] = []
            for e in terms_elements:
                term_id = e.attrib["Id"]
                category = term_id_to_category_dict.get(term_id, "")
                if term_categories and (category == "" or category not in term_categories):
                    continue
                term_id = term_id.replace("\n", "&#xA")
                rendering = e.findtext("Renderings", "")
                renderings = _get_renderings(rendering)
                rows.append(TextRow(text_id, term_id, segment=renderings))
            text = MemoryText(text_id, rows)
            self._add_text(text)


def _get_renderings(rendering: str) -> List[str]:
    # If entire term rendering is surrounded in square brackets, remove them
    match = re.match(r"^\[(.+?)\]$", rendering)
    if match:
        rendering = match.group(1)
    rendering = rendering.replace("?", "")
    rendering = rendering.replace("*", "")
    rendering = rendering.replace("/", " ")
    rendering = rendering.strip()
    rendering = _strip_parens(rendering)
    rendering = _strip_parens(rendering, left="[", right="]")
    rx = re.compile(r"\s+\d+(\.\d+)*$")
    for match in rx.findall(rendering):
        rendering = rendering.replace(match, "")
    glosses = re.split(r"\|\|", rendering)
    glosses = list(set(g.strip() for g in glosses if g.strip() != ""))
    return glosses


def _strip_parens(term_string: str, left: str = "(", right: str = ")") -> str:
    parens = 0
    end = -1
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


def _get_category_per_id(biblical_terms_tree: ElementTree.ElementTree) -> Dict[str, Optional[str]]:
    term_id_to_category_dict = {}
    for e in biblical_terms_tree.iter(".//Term"):
        term_id = e.attrib["Id"]
        if term_id not in term_id_to_category_dict:
            category_element = e.find("Category")
            category = (
                category_element.text if category_element is not None and category_element.text is not None else ""
            )
            term_id_to_category_dict[term_id] = category
    return term_id_to_category_dict
