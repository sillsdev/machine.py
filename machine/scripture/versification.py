from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union

import regex

from .canon import is_canonical
from .verse_ref import VerseRef

NON_CANONICAL_LAST_CHAPTER_OR_VERSE = 998


class Versification:
    def __init__(self, name: str) -> None:
        self._name = name
        self._mappings = _VerseMappings()
        self._excluded_verses: Set[int] = set()
        self._book_list: List[List[int]] = []
        self._verse_segments: Dict[int, List[str]] = {}
        self.description: Optional[str] = None

    def last_book(self) -> int:
        return len(self._book_list)

    def last_chapter(self, book_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self._book_list):
            return 1

        chapters = self._book_list[book_num - 1]
        return len(chapters)

    def last_verse(self, book_num: int, chapter_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self._book_list):
            return 1

        chapters = self._book_list[book_num - 1]
        if chapter_num > len(chapters) or chapter_num < 1:
            return 1

        return chapters[chapter_num - 1]

    def verse_segments(self, bbbcccvvv: int) -> Optional[List[str]]:
        return self._verse_segments.get(bbbcccvvv)

    def change_versification(self, vref: VerseRef) -> None:
        if vref.versification is None:
            vref.versification = self
            return

        orig_versification = vref.versification

        # Map from existing to standard versification

        orig_verse = vref.copy()
        orig_verse.versification = None

        standard_verse = orig_versification._mappings.get_standard(orig_verse)
        if standard_verse is None:
            standard_verse = orig_verse

        # If both versifications contain this verse and map this verse to the same location then no versification
        # change is needed.
        standard_verse_this_versification = self._mappings.get_standard(orig_verse)
        if standard_verse_this_versification is None:
            standard_verse_this_versification = orig_verse

        # ESG is a special case since we have added mappings from verses to LXX segments in several versifications and
        # want this mapping to work both ways.
        if (
            vref.book != "ESG"
            and standard_verse == standard_verse_this_versification
            and self._book_chapter_verse_exists(vref)
        ):
            vref.versification = self
            return

        # Map from standard versification to this versification
        new_verse = self._mappings.get_versification(standard_verse)
        if new_verse is None:
            new_verse = standard_verse

        # If verse has changed, parse new value
        if orig_verse != new_verse:
            vref.copy_from(new_verse)

        vref.versification = self

    def change_versification_with_ranges(self, vref: VerseRef) -> Tuple[bool, VerseRef]:
        parts: List[str] = regex.split(r"[,\-]", vref.verse)

        new_vref = vref.copy()
        new_vref.verse = parts[0]
        self.change_versification(new_vref)
        all_same_chapter = True

        for i in range(2, len(parts)):
            part_vref = vref.copy()
            part_vref.verse = parts[i]
            self.change_versification(part_vref)
            if new_vref.chapter_num != part_vref.chapter_num:
                all_same_chapter = False
            new_vref.verse = new_vref.verse + parts[i - 1] + part_vref.verse

        return all_same_chapter, new_vref

    def __eq__(self, other: "Versification") -> bool:
        if self is other:
            return True
        return (
            self._name == other._name
            and self.description == other.description
            and self._book_list == other._book_list
            and self._excluded_verses == other._excluded_verses
            and self._verse_segments == other._verse_segments
            and self._mappings == other._mappings
        )

    def _book_chapter_verse_exists(self, vref: VerseRef) -> bool:
        return (
            vref.book_num <= self.last_book()
            and vref.chapter_num <= self.last_chapter(vref.book_num)
            and vref.verse_num <= self.last_verse(vref.book_num, vref.chapter_num)
        )


class _VerseMappings:
    def __init__(self) -> None:
        self._versification_to_standard: Dict[VerseRef, VerseRef] = {}
        self._standard_to_versification: Dict[VerseRef, VerseRef] = {}

    def get_versification(self, standard: VerseRef) -> Optional[VerseRef]:
        return self._standard_to_versification.get(standard)

    def get_standard(self, versification: VerseRef) -> Optional[VerseRef]:
        return self._versification_to_standard.get(versification)

    def __eq__(self, other: "_VerseMappings") -> bool:
        return (
            self._versification_to_standard == other._versification_to_standard
            and self._standard_to_versification == other._standard_to_versification
        )


class VersificationType(IntEnum):
    UNKNOWN = 0
    ORIGINAL = 1
    SEPTUAGINT = 2
    VULGATE = 3
    ENGLISH = 4
    RUSSIAN_PROTESTANT = 5
    RUSSIAN_ORTHODOX = 6


def get_versification(type: Union[VersificationType, int]) -> Versification:
    raise RuntimeError("Not implemented.")
