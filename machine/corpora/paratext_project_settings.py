from dataclasses import dataclass
from typing import Optional

from ..scripture.canon import book_id_to_number, book_number_to_id
from ..scripture.verse_ref import Versification
from .usfm_stylesheet import UsfmStylesheet


@dataclass
class ParatextProjectSettings:
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
    language_code: Optional[str]

    def get_book_id(self, file_name: str) -> Optional[str]:
        """Returns None when the file name doesn't match the pattern of a book file name for the project."""
        if not file_name.startswith(self.file_name_prefix) or not file_name.endswith(self.file_name_suffix):
            return None

        book_part: str = file_name[len(self.file_name_prefix) : -len(self.file_name_suffix)]
        if self.file_name_form == "MAT":
            if len(book_part) != 3:
                return None
            book_id = book_part
        elif self.file_name_form in ("40", "41"):
            if book_part != "100" and len(book_part) != 2:
                return None
            book_id = book_number_to_id(_get_book_number(book_part))
        else:
            if book_part.startswith("100"):
                if len(book_part) != 6:
                    return None
            elif len(book_part) != 5:
                return None
            book_id = book_part[2:] if len(book_part) == 5 else book_part[3:]
        return book_id

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


def _get_book_number(book_file_name_digits: str) -> int:
    if book_file_name_digits.startswith("A"):
        return 100 + int(book_file_name_digits[1:])
    if book_file_name_digits.startswith("B"):
        return 110 + int(book_file_name_digits[1:])
    if book_file_name_digits.startswith("C"):
        return 120 + int(book_file_name_digits[1:])

    book_num: int = int(book_file_name_digits)
    if book_num >= 40:
        return book_num - 1
    return book_num
