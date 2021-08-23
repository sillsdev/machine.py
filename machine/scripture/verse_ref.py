from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, TextIO, Tuple, Union

import regex as re

from ..utils.comparable import Comparable
from ..utils.string_utils import is_integer, parse_integer
from ..utils.typeshed import StrPath
from .canon import LAST_BOOK, book_id_to_number, book_number_to_id, is_canonical

VERSE_RANGE_SEPARATOR = "-"
VERSE_SEQUENCE_INDICATOR = ","
_CHAPTER_DIGIT_SHIFTER = 1000
_BOOK_DIGIT_SHIFTER = _CHAPTER_DIGIT_SHIFTER * _CHAPTER_DIGIT_SHIFTER
_BCV_MAX_VALUE = _CHAPTER_DIGIT_SHIFTER


class ValidStatus(Enum):
    VALID = auto()
    UNKNOWN_VERSIFICATION = auto()
    OUT_OF_RANGE = auto()
    VERSE_OUT_OF_ORDER = auto()
    VERSE_REPEATED = auto()


class VerseRef(Comparable):
    def __init__(
        self,
        book: Union[str, int] = 0,
        chapter: Union[str, int] = 0,
        verse: Union[str, int] = 0,
        versification: Optional["Versification"] = None,
    ) -> None:
        if book == 0 and chapter == 0 and verse == 0 and versification is None:
            self._book_num = 0
            self._chapter_num = 0
            self._verse_num = 0
            self._verse = None
            self.versification = NULL_VERSIFICATION
        else:
            if isinstance(book, str):
                self.book_num = book_id_to_number(book)
            else:
                self.book_num = book

            if isinstance(chapter, str):
                self.chapter = chapter
            else:
                self.chapter_num = chapter

            if isinstance(verse, str):
                self.verse = verse
            else:
                self.verse_num = verse

            self.versification = (
                Versification.get_builtin(VersificationType.ENGLISH) if versification is None else versification
            )

    @classmethod
    def from_string(cls, verse_str: str, versification: Optional["Versification"] = None) -> "VerseRef":
        verse_str = verse_str.replace("\u200f", "")
        if verse_str.find("/") >= 0:
            parts = verse_str.split("/")
            verse_str = parts[0]
            if len(parts) > 1:
                type = parse_integer(parts[1].strip())
                if type is None:
                    raise ValueError("The verse reference is invalid.")
                versification = Versification.get_builtin(type)

        b_cv = verse_str.strip().split(" ")
        if len(b_cv) != 2:
            raise ValueError("The verse reference is invalid.")

        c_v = b_cv[1].split(":")

        cnum = parse_integer(c_v[0])
        if (
            len(c_v) != 2
            or book_id_to_number(b_cv[0]) == 0
            or cnum is None
            or cnum < 0
            or not _is_verse_parseable(c_v[1])
        ):
            raise ValueError("The verse reference is invalid.")
        return VerseRef(b_cv[0], c_v[0], c_v[1], versification)

    @classmethod
    def from_range(cls, start: "VerseRef", end: "VerseRef") -> "VerseRef":
        if start.book_num != end.book_num or start.chapter_num != end.chapter_num:
            raise ValueError("The start and end verses are not in the same chapter.")
        if start.has_multiple:
            raise ValueError("This start verse contains multiple verses.")
        if end.has_multiple:
            raise ValueError("This end verse contains multiple verses.")

        return VerseRef(start.book, start.chapter, f"{start.verse_num}-{end.verse_num}")

    @classmethod
    def from_bbbcccvvv(cls, bbbcccvvv: int, versification: Optional["Versification"] = None) -> "VerseRef":
        book = bbbcccvvv // 1000000
        chapter = bbbcccvvv % 1000000 // 1000
        verse = bbbcccvvv % 1000
        return VerseRef(book, chapter, verse, versification)

    @property
    def book_num(self) -> int:
        return self._book_num

    @book_num.setter
    def book_num(self, value: int) -> None:
        if value <= 0 or value > LAST_BOOK:
            raise ValueError("The book number must be greater than zero and less than or equal to the last book.")
        self._book_num = value

    @property
    def chapter_num(self) -> int:
        return self._chapter_num

    @chapter_num.setter
    def chapter_num(self, value: int) -> None:
        if value < 0:
            raise ValueError("The chapter number cannot be negative.")
        self._chapter_num = value

    @property
    def verse_num(self) -> int:
        return self._verse_num

    @verse_num.setter
    def verse_num(self, value: int) -> None:
        if value < 0:
            raise ValueError("The verse number cannot be negative.")
        self._verse_num = value
        self._verse = None

    @property
    def book(self) -> str:
        return book_number_to_id(self.book_num, error_value="")

    @book.setter
    def book(self, value: str) -> None:
        self.book_num = book_id_to_number(value)

    @property
    def chapter(self) -> str:
        return "" if self._chapter_num < 0 else str(self.chapter_num)

    @chapter.setter
    def chapter(self, value: str) -> None:
        chapter_num = parse_integer(value)
        if chapter_num is None:
            chapter_num = -1
        self._chapter_num = chapter_num

    @property
    def verse(self) -> str:
        if self._verse is not None:
            return self._verse
        return "" if self._verse_num < 0 else str(self._verse_num)

    @verse.setter
    def verse(self, value: str) -> None:
        result, self._verse_num = _get_verse_num(value)
        self._verse = None if result else value.replace("\u200f", "")
        if self._verse_num < 0:
            _, self._verse_num = _get_verse_num(self._verse)

    @property
    def bbbcccvvv(self) -> int:
        return get_bbbcccvvv(self._book_num, self._chapter_num, self._verse_num)

    @property
    def bbbcccvvvs(self) -> str:
        return str(self.bbbcccvvv).zfill(9) + self.segment()

    @property
    def has_multiple(self) -> bool:
        return self._verse is not None and (
            self._verse.find(VERSE_RANGE_SEPARATOR) != -1 or self._verse.find(VERSE_SEQUENCE_INDICATOR) != -1
        )

    @property
    def valid_status(self) -> ValidStatus:
        if self._verse is None or self._verse == "":
            return self._validate_single_verse()

        prev_verse = 0
        for vref in self.all_verses():
            valid_status = vref._validate_single_verse()
            if valid_status != ValidStatus.VALID:
                return valid_status

            bbbcccvvv = vref.bbbcccvvv
            if prev_verse > bbbcccvvv:
                return ValidStatus.VERSE_OUT_OF_ORDER
            if prev_verse == bbbcccvvv:
                return ValidStatus.VERSE_REPEATED
            prev_verse = bbbcccvvv
        return ValidStatus.VALID

    @property
    def is_valid(self) -> bool:
        return self.valid_status == ValidStatus.VALID

    @property
    def is_excluded(self) -> bool:
        return self.versification.is_excluded(self.bbbcccvvv)

    def get_segments(self, default_segments: Optional[Set[str]] = None) -> Optional[Set[str]]:
        if self.versification is None:
            return default_segments

        segments = self.versification.verse_segments.get(self.bbbcccvvv)
        if segments is None:
            segments = default_segments
        return segments

    def validated_segment(self, valid_segments: Optional[Set[str]] = None) -> str:
        seg = self.segment()
        if len(seg) == 0:
            return ""

        valid_segments = self.get_segments(valid_segments)
        if valid_segments is not None and len(valid_segments) > 0:
            return seg if seg in valid_segments else ""

        return seg

    def segment(self) -> str:
        if self._verse is None or self._verse == "" or not is_integer(self._verse[0]):
            return ""

        found_seg_start = False
        seg = ""
        for c in self._verse:
            if c == VERSE_RANGE_SEPARATOR or c == VERSE_SEQUENCE_INDICATOR:
                break

            if not is_integer(c):
                found_seg_start = True
                seg += c
            elif found_seg_start:
                break

        return seg

    def simplify(self) -> None:
        self._verse = None

    def all_verses(self) -> Iterable["VerseRef"]:
        if self._verse is None or self.chapter_num <= 0:
            yield self.copy()
        else:
            parts = self._verse.split(VERSE_SEQUENCE_INDICATOR)
            for part in parts:
                pieces = part.split(VERSE_RANGE_SEPARATOR)
                start_verse = self.copy()
                start_verse.verse = pieces[0]
                yield start_verse

                if len(pieces) > 1:
                    last_verse = self.copy()
                    last_verse.verse = pieces[1]
                    for verse_num in range(start_verse.verse_num + 1, last_verse.verse_num):
                        verse_in_range = VerseRef(self.book_num, self.chapter_num, verse_num, self.versification)
                        if not verse_in_range.is_excluded:
                            yield verse_in_range
                    yield last_verse

    def unbridge(self) -> "VerseRef":
        return next(iter(self.all_verses()))

    def get_ranges(self) -> Iterable["VerseRef"]:
        if self._verse is None or self.chapter_num <= 0:
            yield self.copy()
        else:
            ranges = self._verse.split(",")
            for range in ranges:
                vref = self.copy()
                vref.verse = range
                yield vref

    def copy(self) -> "VerseRef":
        copy = VerseRef()
        copy._book_num = self._book_num
        copy._chapter_num = self._chapter_num
        copy._verse_num = self._verse_num
        copy._verse = self._verse
        copy.versification = self.versification
        return copy

    def copy_from(self, vref: "VerseRef") -> None:
        self._book_num = vref._book_num
        self._chapter_num = vref._chapter_num
        self._verse_num = vref._verse_num
        self._verse = vref._verse
        self.versification = vref.versification

    def copy_verse_from(self, vref: "VerseRef") -> None:
        self._verse_num = vref._verse_num
        self._verse = vref._verse

    def change_versification(self, versification: "Versification") -> bool:
        return versification.change_versification(self)

    def str_with_versification(self) -> str:
        return f"{str(self)}/{int(self.versification.type)}"

    def compare_to(self, other: object, compare_all_verses: bool = True) -> int:
        if not isinstance(other, VerseRef):
            raise TypeError("other is not a VerseRef object.")
        if self is other:
            return 0

        if self.versification != other.versification:
            other = other.copy()
            other.change_versification(self.versification)

        if self.book_num != other.book_num:
            return self.book_num - other.book_num
        if self.chapter_num != other.chapter_num:
            return self.chapter_num - other.chapter_num
        if compare_all_verses:
            # compare all available verses (whether a single verse or a verse bridge)
            return self._compare_verses(other)

        # compare only the first verse bridge
        if self.verse_num != other.verse_num:
            return self.verse_num - other.verse_num

        this_segment = self.validated_segment()
        other_segment = other.validated_segment()
        if this_segment == "" and other_segment == "":
            return 0
        if this_segment == "" and other_segment != "":
            return -1
        if this_segment != "" and other_segment == "":
            return 1

        if this_segment < other_segment:
            return -1
        if this_segment > other_segment:
            return 1
        return 0

    def __hash__(self) -> int:
        if self._verse is not None:
            return self.bbbcccvvv ^ hash(self._verse)
        return self.bbbcccvvv

    def __repr__(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def _compare_verses(self, other: "VerseRef") -> int:
        verse_list = list(self.all_verses())
        other_verse_list = list(other.all_verses())

        for verse, other_verse in zip(verse_list, other_verse_list):
            result = verse.compare_to(other_verse, compare_all_verses=False)
            if result != 0:
                return result
        return len(verse_list) - len(other_verse_list)

    def _validate_single_verse(self) -> ValidStatus:
        # Unknown versification is always invalid
        if self.versification is None:
            return ValidStatus.UNKNOWN_VERSIFICATION

        # If invalid book, reference is invalid
        if self._book_num <= 0 or self._book_num > LAST_BOOK:
            return ValidStatus.OUT_OF_RANGE

        # If non-biblical book, any chapter/verse is valid
        if not is_canonical(self._book_num):
            return ValidStatus.VALID

        if (
            self._book_num > self.versification.get_last_book()
            or self._chapter_num <= 0
            or self._chapter_num > self.versification.get_last_chapter(self._book_num)
            or self.verse_num < 0
            or self.verse_num > self.versification.get_last_verse(self._book_num, self._chapter_num)
        ):
            return ValidStatus.OUT_OF_RANGE

        return ValidStatus.OUT_OF_RANGE if self.versification.is_excluded(self.bbbcccvvv) else ValidStatus.VALID


def get_bbbcccvvv(book_num: int, chapter_num: int, verse_num: int) -> int:
    return (
        (book_num % _BCV_MAX_VALUE) * _BOOK_DIGIT_SHIFTER
        + ((chapter_num % _BCV_MAX_VALUE) * _CHAPTER_DIGIT_SHIFTER if chapter_num >= 0 else 0)
        + (verse_num % _BCV_MAX_VALUE if verse_num >= 0 else 0)
    )


def are_overlapping_verse_ranges(verse1: str, verse2: str) -> bool:
    verse1_parts = verse1.split(VERSE_SEQUENCE_INDICATOR)
    verse2_parts = verse2.split(VERSE_SEQUENCE_INDICATOR)

    for verse1_part in verse1_parts:
        for verse2_part in verse2_parts:
            verse1_num, verse1_seg, verse1_end_num, verse1_end_seg = _parse_verse_number_range(verse1_part)
            verse2_num, verse2_seg, verse2_end_num, verse2_end_seg = _parse_verse_number_range(verse2_part)

            if (
                verse1_num == verse1_end_num
                and verse2_num == verse2_end_num
                and verse1_seg == verse1_end_seg
                and verse2_seg == verse1_end_seg
            ):
                # no ranges, this is easy
                if verse1_num == verse2_num and (verse1_seg == "" or verse2_seg == "" or verse1_seg == verse2_seg):
                    return True
            elif _in_verse_range(verse1_num, verse1_seg, verse2_num, verse2_seg, verse2_end_num, verse2_end_seg):
                return True
            elif _in_verse_range(
                verse1_end_num, verse1_end_seg, verse2_num, verse2_seg, verse2_end_num, verse2_end_seg
            ):
                return True
            elif _in_verse_range(verse2_num, verse2_seg, verse1_num, verse1_seg, verse1_end_num, verse1_end_seg):
                return True
            elif _in_verse_range(
                verse2_end_num, verse2_end_seg, verse1_num, verse1_seg, verse1_end_num, verse1_end_seg
            ):
                return True
    return False


def _in_verse_range(
    verse1: int, verse1_seg: str, verse2: int, verse2_seg: str, verse2_end: int, verse2_end_seg: str
) -> bool:
    if verse1 < verse2:
        return False

    if verse1 == verse2 and verse1_seg != "" and verse2_seg != "":
        if verse1_seg < verse2_seg:
            return False

    if verse1 > verse2_end:
        return False

    if verse1 == verse2_end and verse1_seg != "" and verse2_end_seg != "":
        if verse1_seg > verse2_end_seg:
            return False
    return True


def _parse_verse_number_range(v_num: str) -> Tuple[int, str, int, str]:
    parts = re.split(r"[\-\u2013\u2014]", v_num)
    if len(parts) == 1:
        number, segment = _parse_verse_number(parts[0])
        return number, segment, number, segment

    number1, segment1 = _parse_verse_number(parts[0])
    number2, segment2 = _parse_verse_number(parts[1])
    return number1, segment1, number2, segment2


def _parse_verse_number(v_num: str) -> Tuple[int, str]:
    j = 0
    while j < len(v_num) and v_num[j].isdigit():
        j += 1

    number = 0
    if j > 0:
        num = v_num[:j]
        number = int(num)
    return number, v_num[j:]


def _is_verse_parseable(verse: str) -> bool:
    return (
        len(verse) != 0
        and is_integer(verse[0])
        and verse[-1] != VERSE_RANGE_SEPARATOR
        and verse[-1] != VERSE_SEQUENCE_INDICATOR
    )


def _get_verse_num(verse: Optional[str]) -> Tuple[bool, int]:
    if verse is None or verse == "":
        return True, -1

    v_num = 0
    for i in range(len(verse)):
        ch = verse[i]
        if not is_integer(ch):
            if i == 0:
                v_num = -1
            return False, v_num

        v_num = (v_num * 10) + int(ch)
        if v_num > 999:
            v_num = -1
            return False, v_num
    return True, v_num


class VersificationType(IntEnum):
    UNKNOWN = 0
    ORIGINAL = 1
    SEPTUAGINT = 2
    VULGATE = 3
    ENGLISH = 4
    RUSSIAN_PROTESTANT = 5
    RUSSIAN_ORTHODOX = 6


class Versification:
    _BUILTIN_VERSIFICATIONS: Dict[VersificationType, "Versification"] = {}
    _BUILTIN_VERSIFICATION_FILENAMES = {
        VersificationType.ORIGINAL: "org.vrs.txt",
        VersificationType.ENGLISH: "eng.vrs.txt",
        VersificationType.SEPTUAGINT: "lxx.vrs.txt",
        VersificationType.VULGATE: "vul.vrs.txt",
        VersificationType.RUSSIAN_ORTHODOX: "rso.vrs.txt",
        VersificationType.RUSSIAN_PROTESTANT: "rsc.vrs.txt",
    }
    _BUILTIN_VERSIFICATION_NAMES_TO_TYPES = {
        "Original": VersificationType.ORIGINAL,
        "Septuagint": VersificationType.SEPTUAGINT,
        "Vulgate": VersificationType.VULGATE,
        "English": VersificationType.ENGLISH,
        "RussianProtestant": VersificationType.RUSSIAN_PROTESTANT,
        "RussianOrthodox": VersificationType.RUSSIAN_ORTHODOX,
    }
    _NON_CANONICAL_LAST_CHAPTER_OR_VERSE = 998

    @classmethod
    def create(cls, name: str) -> "Versification":
        type = cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES.get(name)
        if type is not None:
            return cls.get_builtin(type)

        versification = Versification(name)
        with open(Path(__file__).parent / "eng.vrs.txt", "r", encoding="utf-8-sig") as file:
            versification = cls.parse(file, versification=versification)
        return versification

    @classmethod
    def get_builtin(cls, type: Union[VersificationType, int, str]) -> "Versification":
        if isinstance(type, int):
            type = VersificationType(type)
        else:
            if type in cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES:
                type = cls._BUILTIN_VERSIFICATION_NAMES_TO_TYPES[type]
            else:
                type = VersificationType[type]

        versification = cls._BUILTIN_VERSIFICATIONS.get(type)
        if versification is not None:
            return versification

        filename = cls._BUILTIN_VERSIFICATION_FILENAMES.get(type)
        if filename is None:
            raise ValueError("The versification type is unknown.")

        path = Path(__file__).parent / filename

        with open(path, "r", encoding="utf-8-sig") as file:
            versification = cls.parse(file)
        cls._BUILTIN_VERSIFICATIONS[type] = versification
        return versification

    @classmethod
    def load(
        cls,
        filename: StrPath,
        base_versification: Optional["Versification"] = None,
        fallback_name: Optional[str] = None,
    ) -> "Versification":
        with open(filename, "r", encoding="utf-8-sig") as file:
            versification = (
                None
                if base_versification is None or fallback_name is None
                else Versification(fallback_name, filename, base_versification)
            )
            return cls.parse(file, filename, versification, fallback_name)

    @classmethod
    def parse(
        cls,
        stream: TextIO,
        filename: Optional[StrPath] = None,
        versification: Optional["Versification"] = None,
        fallback_name: Optional[str] = None,
    ) -> "Versification":
        return _parse_versification(stream, filename, versification, fallback_name)

    def __init__(
        self, name: str, filename: Optional[StrPath] = None, base_versification: Optional["Versification"] = None
    ) -> None:
        self._name = name
        self._type = VersificationType.UNKNOWN
        if base_versification is None:
            type = Versification._BUILTIN_VERSIFICATION_NAMES_TO_TYPES.get(name)
            if type is not None:
                self._type = type
        self._filename = None if filename is None else Path(filename)
        self._base_versification = base_versification

        self.mappings = VerseMappings()
        self.excluded_verses: Set[int] = set()
        self.book_list: List[List[int]] = []
        self.verse_segments: Dict[int, Set[str]] = {}
        self.description: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def filename(self) -> Optional[Path]:
        return self._filename

    @property
    def type(self) -> "VersificationType":
        return self._type

    @property
    def base_versification(self) -> Optional["Versification"]:
        return self._base_versification

    def get_last_book(self) -> int:
        return len(self.book_list)

    def get_last_chapter(self, book_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return Versification._NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self.book_list):
            return 1

        chapters = self.book_list[book_num - 1]
        return len(chapters)

    def get_last_verse(self, book_num: int, chapter_num: int) -> int:
        # Non-scripture books have 998 chapters.
        # Use 998 so the VerseRef.BBBCCCVVV value is computed properly.
        if not is_canonical(book_num):
            return Versification._NON_CANONICAL_LAST_CHAPTER_OR_VERSE

        # Anything else not in .vrs file has 1 chapter
        if book_num > len(self.book_list):
            return 1

        chapters = self.book_list[book_num - 1]
        if chapter_num > len(chapters) or chapter_num < 1:
            return 1

        return chapters[chapter_num - 1]

    def first_included_verse(self, book_num: int, chapter_num: int) -> Optional[VerseRef]:
        while True:
            last_verse = self.get_last_verse(book_num, chapter_num)
            for verse_num in range(1, last_verse + 1):
                if not self.is_excluded(get_bbbcccvvv(book_num, chapter_num, verse_num)):
                    return VerseRef(book_num, chapter_num, verse_num, self)

            # Non-excluded verse not found in this chapter, so try next chapter
            chapter_num += 1
            if chapter_num > self.get_last_chapter(book_num):
                break

        return None

    def is_excluded(self, bbbcccvvv: int) -> bool:
        return bbbcccvvv in self.excluded_verses

    def change_versification(self, vref: VerseRef) -> bool:
        if vref.has_multiple:
            return self._change_versification_with_ranges(vref)

        if vref.versification == NULL_VERSIFICATION:
            vref.versification = self
            return True

        orig_versification = vref.versification

        # Map from existing to standard versification
        orig_vref = vref.copy()
        orig_vref.versification = NULL_VERSIFICATION

        standard_vref = orig_versification.mappings.get_standard(orig_vref)
        if standard_vref is None:
            standard_vref = orig_vref

        # If both versifications contain this verse and map this verse to the same location then no versification
        # change is needed.
        standard_vref_this_versification = self.mappings.get_standard(orig_vref)
        if standard_vref_this_versification is None:
            standard_vref_this_versification = orig_vref

        # ESG is a special case since we have added mappings from verses to LXX segments in several versifications and
        # want this mapping to work both ways.
        if (
            vref.book != "ESG"
            and standard_vref == standard_vref_this_versification
            and self._book_chapter_verse_exists(vref)
        ):
            vref.versification = self
            return True

        # Map from standard versification to this versification
        new_vref = self.mappings.get_versification(standard_vref)
        if new_vref is None:
            new_vref = standard_vref

        # If verse has changed, parse new value
        if orig_vref != new_vref:
            vref.copy_from(new_vref)

        vref.versification = self
        return True

    def __eq__(self, other: "Versification") -> bool:
        if self is other:
            return True
        return (
            self._name == other._name
            and self.description == other.description
            and self.book_list == other.book_list
            and self.excluded_verses == other.excluded_verses
            and self.verse_segments == other.verse_segments
            and self.mappings == other.mappings
        )

    def _book_chapter_verse_exists(self, vref: VerseRef) -> bool:
        return (
            vref.book_num <= self.get_last_book()
            and vref.chapter_num <= self.get_last_chapter(vref.book_num)
            and vref.verse_num <= self.get_last_verse(vref.book_num, vref.chapter_num)
        )

    def _change_versification_with_ranges(self, vref: VerseRef) -> bool:
        parts: List[str] = re.split(r"([,\-])", vref.verse)

        new_vref = vref.copy()
        new_vref.verse = parts[0]
        self.change_versification(new_vref)
        all_same_chapter = True

        for i in range(2, len(parts), 2):
            part_vref = vref.copy()
            part_vref.verse = parts[i]
            self.change_versification(part_vref)
            if new_vref.chapter_num != part_vref.chapter_num:
                all_same_chapter = False
            new_vref.verse = new_vref.verse + parts[i - 1] + part_vref.verse

        vref.copy_from(new_vref)

        return all_same_chapter


class VerseMappings:
    def __init__(self) -> None:
        self._versification_to_standard: Dict[VerseRef, VerseRef] = {}
        self._standard_to_versification: Dict[VerseRef, VerseRef] = {}

    def add_mapping(self, versification_ref: VerseRef, standard_ref: VerseRef) -> None:
        if sum(1 for _ in versification_ref.all_verses()) != 1 or sum(1 for _ in standard_ref.all_verses()) != 1:
            raise ValueError("Mappings must resolve into a single reference on both sides.")

        self._versification_to_standard[versification_ref] = standard_ref
        self._standard_to_versification[standard_ref] = versification_ref

    def add_mappings(self, versification_refs: List[VerseRef], standard_refs: List[VerseRef]) -> None:
        for versification_ref in versification_refs:
            for standard_ref in standard_refs:
                self.add_mapping(versification_ref, standard_ref)

    def get_versification(self, standard_ref: VerseRef) -> Optional[VerseRef]:
        return self._standard_to_versification.get(standard_ref)

    def get_standard(self, versification_ref: VerseRef) -> Optional[VerseRef]:
        return self._versification_to_standard.get(versification_ref)

    def __eq__(self, other: "VerseMappings") -> bool:
        return (
            self._versification_to_standard == other._versification_to_standard
            and self._standard_to_versification == other._standard_to_versification
        )


NULL_VERSIFICATION = Versification("NULL")

_VERSIFICATION_NAME_REGEX = re.compile(r"#\s*Versification\s+\"(?<name>[^\"]+)\"\s*")


class _LineType(Enum):
    COMMENT = auto()
    CHAPTER_VERSE = auto()
    STANDARD_MAPPING = auto()
    ONE_TO_MANY_MAPPING = auto()
    EXCLUDED_VERSE = auto()
    VERSE_SEGMENTS = auto()


@dataclass(frozen=True)
class _VersificationLine:
    type: _LineType
    line: str
    comment: str
    line_num: int

    def __repr__(self) -> str:
        if self.type == _LineType.CHAPTER_VERSE:
            return self.line
        elif self.type == _LineType.ONE_TO_MANY_MAPPING:
            return f"#! {self.line}"
        elif self.type == _LineType.COMMENT:
            if self.comment != "":
                return f"# {self.comment}"
            return ""
        else:
            if self.comment == "":
                return self.line
            return f"{self.line} # {self.comment}"


def _syntax_error(message: str, line_num: int) -> RuntimeError:
    return RuntimeError(f"Invalid versification syntax: {message}, line: {line_num}.")


def _parse_versification(
    stream: TextIO,
    filename: Optional[StrPath] = None,
    versification: Optional["Versification"] = None,
    fallback_name: Optional[str] = None,
) -> Versification:
    line_num = 1
    for line in stream:
        line = line.strip()
        if versification is None:
            name = ""
            match = _VERSIFICATION_NAME_REGEX.match(line)
            if match is not None:
                name = match.group("name")
            if name != "":
                versification = Versification(name, filename)

        parsed_line = _parse_line(line, line_num)
        if parsed_line.type == _LineType.COMMENT:
            line_num += 1
            continue

        if versification is None:
            if fallback_name is None or fallback_name == "":
                raise _syntax_error("the versification is missing a name", parsed_line.line_num)
            versification = Versification(fallback_name, filename)

        if parsed_line.type == _LineType.CHAPTER_VERSE:
            _parse_chapter_verse_line(versification, parsed_line)
        elif parsed_line.type == _LineType.STANDARD_MAPPING:
            _parse_mapping_line(versification, parsed_line)
        elif parsed_line.type == _LineType.ONE_TO_MANY_MAPPING:
            _parse_range_to_one_mapping_line(versification, parsed_line)
        elif parsed_line.type == _LineType.EXCLUDED_VERSE:
            _parse_excluded_verse(versification, parsed_line)
        elif parsed_line.type == _LineType.VERSE_SEGMENTS:
            if parsed_line.line.find("#") != -1:
                raise _syntax_error("invalid verse segments line", parsed_line.line_num)
            _parse_verse_segments_line(versification, parsed_line, is_builtin=filename is None)
        line_num += 1

    assert versification is not None
    return versification


def _parse_line(line: str, line_num: int) -> _VersificationLine:
    is_comment_line = len(line) > 0 and line[0] == "#"
    parts = line.split("#", maxsplit=2)
    line = parts[0].strip()
    comment = parts[1].strip() if len(parts) == 2 else ""

    if line == "" and len(comment) > 2 and comment[0] == "!":
        # found Paratext 7.3(+) versification line beginning with #!
        line = comment[1:].strip()
        comment = ""
        is_comment_line = False

    if len(line) == 0 or is_comment_line:
        line_type = _LineType.COMMENT
    elif "=" in line:
        # mapping one verse to multiple
        line_type = _LineType.ONE_TO_MANY_MAPPING if line[0] == "&" else _LineType.STANDARD_MAPPING
    elif line[0] == "-":
        line_type = _LineType.EXCLUDED_VERSE
    elif line[0] == "*":
        line_type = _LineType.VERSE_SEGMENTS
    else:
        line_type = _LineType.CHAPTER_VERSE

    return _VersificationLine(line_type, line, comment, line_num)


def _parse_chapter_verse_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    parts = parsed_line.line.split()
    book_num = book_id_to_number(parts[0])
    if book_num == 0:
        raise _syntax_error("invalid book", parsed_line.line_num)

    while len(versification.book_list) < book_num:
        versification.book_list.append([1])

    verses_in_chapter = versification.book_list[book_num - 1].copy()

    chapter = 0
    for part in parts[1:]:
        # END is used if the number of chapters in custom is less than base
        if part == "END":
            if len(verses_in_chapter) > chapter:
                del verses_in_chapter[chapter:]
            break

        pieces = part.split(":")
        c = parse_integer(pieces[0])
        if c is None or c <= 0:
            raise _syntax_error("invalid chapter", parsed_line.line_num)
        chapter = c

        if len(pieces) != 2:
            raise _syntax_error("missing verse", parsed_line.line_num)

        verse_count = parse_integer(pieces[1])
        if verse_count is None or verse_count < 0:
            raise _syntax_error("invalid verse", parsed_line.line_num)

        if len(verses_in_chapter) < chapter:
            for _ in range(len(verses_in_chapter), chapter):
                # by default, chapters have one verse
                verses_in_chapter.append(1)
        verses_in_chapter[chapter - 1] = verse_count

    versification.book_list[book_num - 1] = verses_in_chapter


def _parse_mapping_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    try:
        parts = parsed_line.line.split("=")
        left_pieces = parts[0].strip().split("-")
        right_pieces = parts[1].strip().split("-")

        new_verse_ref = VerseRef.from_string(left_pieces[0], NULL_VERSIFICATION)
        left_limit = 0 if len(left_pieces) == 1 else int(left_pieces[1])

        standard_verse_ref = VerseRef.from_string(right_pieces[0], NULL_VERSIFICATION)

        while True:
            versification.mappings.add_mapping(new_verse_ref.copy(), standard_verse_ref.copy())

            if new_verse_ref.verse_num >= left_limit:
                break

            new_verse_ref.verse_num += 1
            standard_verse_ref.verse_num += 1
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)


def _get_references(verse_pieces: List[str]) -> List[VerseRef]:
    if len(verse_pieces) == 1:
        return [VerseRef.from_string(verse_pieces[0], NULL_VERSIFICATION)]

    new_verse_ref = VerseRef.from_string(verse_pieces[0], NULL_VERSIFICATION)
    limit = int(verse_pieces[1])

    verse_refs: List[VerseRef] = []
    while True:
        verse_refs.append(new_verse_ref.copy())
        if new_verse_ref.verse_num >= limit:
            break

        new_verse_ref.verse_num += 1

    return verse_refs


def _parse_range_to_one_mapping_line(versification: Versification, parsed_line: _VersificationLine) -> None:
    line = parsed_line.line[1:]
    try:
        parts = line.split("=")
        left_pieces = parts[0].strip().split("-")
        right_pieces = parts[1].strip().split("-")

        versification_refs = _get_references(left_pieces)
        standard_refs = _get_references(right_pieces)
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)

    # either versification or standard must have just one verse
    if len(versification_refs) != 1 and len(standard_refs) != 1:
        raise _syntax_error("invalid many-to-one mapping", parsed_line.line_num)

    versification.mappings.add_mappings(versification_refs, standard_refs)


def _get_verse_reference(parts: List[str], line_num: int) -> Tuple[str, int, int]:
    book_name = parts[0][1:]
    if book_id_to_number(book_name) == 0:
        raise _syntax_error("invalid book", line_num)

    pieces = parts[1].split(":")
    chapter = int(pieces[0])
    verse = int(pieces[1])
    return book_name, chapter, verse


def _parse_excluded_verse(versification: Versification, parsed_line: _VersificationLine) -> None:
    if (
        len(parsed_line.line) < 8
        or parsed_line.line[0] != "-"
        or ":" not in parsed_line.line
        or " " not in parsed_line.line
    ):
        raise _syntax_error("invalid excluded verse line", parsed_line.line_num)

    parts = parsed_line.line.split()
    try:
        book_name, chapter, verse = _get_verse_reference(parts, parsed_line.line_num)

        verse_ref = VerseRef(book_name, str(chapter), str(verse), versification)
        if verse_ref.bbbcccvvv not in versification.excluded_verses:
            versification.excluded_verses.add(verse_ref.bbbcccvvv)
        else:
            raise _syntax_error("duplicate excluded verse", parsed_line.line_num)
    except ValueError:
        raise _syntax_error("invalid verse reference", parsed_line.line_num)


def _remove_spaces(line: str, index: int) -> str:
    if index < 1:
        raise ValueError("Invalid length.")
    if len(line) < 2:
        raise ValueError("Invalid line.")

    result = line[:index]
    parts = line[index:].split()

    for part in parts:
        result += part
    return result


def _parse_verse_segments_line(versification: Versification, parsed_line: _VersificationLine, is_builtin: bool) -> None:
    if (
        len(parsed_line.line) < 8
        or parsed_line.line[0] != "*"
        or ":" not in parsed_line.line
        or " " not in parsed_line.line
        or "," not in parsed_line.line
    ):
        raise _syntax_error("invalid verse segments line", parsed_line.line_num)

    index_of_colon = parsed_line.line.find(":")
    line = _remove_spaces(parsed_line.line, index_of_colon)

    parts = line.split()
    try:
        # Get segmenting information
        segment_start = parts[1].find(",")
        if segment_start == -1:
            raise _syntax_error("invalid segment", parsed_line.line_num)

        segments = parts[1][segment_start + 1 :]

        # Get Scripture reference, throwing an exception if it is not valid.
        parts[1] = parts[1][:segment_start]
        # Remove segment info from chapter:verse reference
        book_name, chapter, verse = _get_verse_reference(parts, parsed_line.line_num)

        segment_set: Set[str] = set()
        nonempty_segment_found = False
        for seg in segments.split(","):
            if seg == "":
                continue
            if nonempty_segment_found and seg == "-":
                raise _syntax_error("unspecified segment location", parsed_line.line_num)

            if seg == "-":
                # '-' indicates no marking for segment
                segment_set.add("")
            else:
                segment_set.add(seg)
                nonempty_segment_found = True

        if len(segment_set) == 1 and next(iter(segment_set)) == "":
            raise _syntax_error("no segments defined", parsed_line.line_num)

        bbbcccvvv = get_bbbcccvvv(book_id_to_number(book_name), chapter, verse)
        # Don't allow overwrites for built-in versifications
        if is_builtin and bbbcccvvv in versification.verse_segments:
            raise _syntax_error("duplicate segment", parsed_line.line_num)

        versification.verse_segments[bbbcccvvv] = segment_set
    except ValueError as e:
        raise _syntax_error("invalid verse reference " + str(e), parsed_line.line_num)
