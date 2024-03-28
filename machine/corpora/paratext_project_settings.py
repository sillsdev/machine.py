from abc import ABC
from dataclasses import dataclass

from ..scripture.canon import book_id_to_number
from ..scripture.verse_ref import Versification
from .usfm_stylesheet import UsfmStylesheet


@dataclass
class ParatextProjectSettings(ABC):
    name: str
    full_name: str
    encoding: str
    versification: Versification
    stylesheet: UsfmStylesheet
    file_name_prefix: str
    file_name_form: str
    file_name_suffix: str
    biblical_terms_list_type: str
    biblical_terms_project_name: str
    biblical_terms_file_name: str

    def get_book_file_name(self, book_id: str) -> str:
        if self.file_name_form == "MAT":
            book_part = book_id
        elif self.file_name_form in ("40", "41"):
            book_part = _get_book_file_name_digits(book_id)
        else:
            book_part = _get_book_file_name_digits(book_id) + book_id
        return self.file_name_prefix + book_part + self.file_name_suffix


def _get_book_file_name_digits(book_id: str) -> str:
    book_num = book_id_to_number(book_id)
    if book_num < 10:
        return f"0{book_num}"
    if book_num < 40:
        return str(book_num)
    if book_num < 100:
        return str(book_num + 1)
    if book_num < 110:
        return f"A{book_num - 100}"
    if book_num < 120:
        return f"B{book_num - 110}"
    return f"C{book_num - 120}"
