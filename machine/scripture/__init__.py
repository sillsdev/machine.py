from typing import Any

from .canon import (
    ALL_BOOK_IDS,
    BOOK_NUMBERS,
    FIRST_BOOK,
    LAST_BOOK,
    NON_CANONICAL_IDS,
    book_id_to_number,
    book_number_to_id,
    get_books,
    is_book_id_valid,
    is_canonical,
    is_nt,
    is_ot,
    is_ot_nt,
)
from .verse_ref import (
    NULL_VERSIFICATION,
    VERSE_RANGE_SEPARATOR,
    VERSE_SEQUENCE_INDICATOR,
    ValidStatus,
    VerseRef,
    Versification,
    VersificationType,
    are_overlapping_verse_ranges,
    get_bbbcccvvv,
)

ORIGINAL_VERSIFICATION: Versification
ENGLISH_VERSIFICATION: Versification
SEPTUAGINT_VERSIFICATION: Versification
VULGATE_VERSIFICATION: Versification
RUSSIAN_ORTHODOX_VERSIFICATION: Versification
RUSSIAN_PROTESTANT_VERSIFICATION: Versification


def __getattr__(name: str) -> Any:
    if name.endswith("_VERSIFICATION"):
        index = name.rindex("_")
        return Versification.get_builtin(name[:index])
    raise AttributeError


__all__ = [
    "ALL_BOOK_IDS",
    "BOOK_NUMBERS",
    "ENGLISH_VERSIFICATION",
    "FIRST_BOOK",
    "LAST_BOOK",
    "NON_CANONICAL_IDS",
    "NULL_VERSIFICATION",
    "ORIGINAL_VERSIFICATION",
    "RUSSIAN_ORTHODOX_VERSIFICATION",
    "RUSSIAN_PROTESTANT_VERSIFICATION",
    "SEPTUAGINT_VERSIFICATION",
    "VERSE_RANGE_SEPARATOR",
    "VERSE_SEQUENCE_INDICATOR",
    "VULGATE_VERSIFICATION",
    "ValidStatus",
    "VerseRef",
    "Versification",
    "VersificationType",
    "are_overlapping_verse_ranges",
    "book_id_to_number",
    "book_number_to_id",
    "get_bbbcccvvv",
    "get_books",
    "is_book_id_valid",
    "is_canonical",
    "is_nt",
    "is_ot",
    "is_ot_nt",
]
