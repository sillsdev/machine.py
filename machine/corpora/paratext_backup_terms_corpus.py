from typing import List, Sequence, Tuple
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .dictionary_text_corpus import DictionaryTextCorpus
from .key_term_row import KeyTerm
from .memory_text import MemoryText
from .text_row import TextRow
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser
from .zip_paratext_project_terms_parser import ZipParatextProjectTermsParser


class ParatextBackupTermsCorpus(DictionaryTextCorpus):
    def __init__(self, filename: StrPath, term_categories: Sequence[str], use_term_glosses: bool = True) -> None:
        super().__init__()

        with ZipFile(filename, "r") as archive:
            settings = ZipParatextProjectSettingsParser(archive).parse()
            key_terms: Sequence[KeyTerm] = ZipParatextProjectTermsParser(archive, settings).parse(
                term_categories, use_term_glosses
            )
            text_id = (
                f"{settings.biblical_terms_list_type}:"
                f"{settings.biblical_terms_project_name}:"
                f"{settings.biblical_terms_file_name}"
            )

            text = MemoryText(text_id, [TextRow(text_id, key_term.id, key_term.renderings) for key_term in key_terms])
            self._add_text(text)
