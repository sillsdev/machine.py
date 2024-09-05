from typing import List, Sequence, Tuple
from zipfile import ZipFile

from ..utils.typeshed import StrPath
from .dictionary_text_corpus import DictionaryTextCorpus
from .memory_text import MemoryText
from .text_row import TextRow
from .zip_paratext_project_settings_parser import ZipParatextProjectSettingsParser
from .zip_paratext_project_terms_parser import ZipParatextProjectTermsParser


class ParatextBackupTermsCorpus(DictionaryTextCorpus):
    def __init__(self, filename: StrPath, term_categories: Sequence[str], use_term_glosses: bool = True) -> None:
        super().__init__()

        with ZipFile(filename, "r") as archive:
            settings = ZipParatextProjectSettingsParser(archive).parse()
            glosses: List[Tuple[str, List[str]]] = ZipParatextProjectTermsParser(archive, settings).parse(
                term_categories, use_term_glosses
            )
            text_id = (
                f"{settings.biblical_terms_list_type}:"
                f"{settings.biblical_terms_project_name}:"
                f"{settings.biblical_terms_file_name}"
            )

            text = MemoryText(text_id, [TextRow(text_id, kvp[0], kvp[1]) for kvp in glosses])
            self._add_text(text)
