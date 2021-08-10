from enum import Enum, auto
from typing import TYPE_CHECKING, Iterable, Optional, Set, Tuple, Union

import regex

from ..utils.comparable import Comparable
from ..utils.string_utils import is_integer, parse_integer
from .canon import LAST_BOOK, book_id_to_number, book_number_to_id, is_canonical

if TYPE_CHECKING:
    from .versification import Versification

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
        from .versification import NULL_VERSIFICATION, Versification, VersificationType

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
        from .versification import Versification

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

        segments = self.versification.verse_segments(self.bbbcccvvv)
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
    parts = regex.split(r"[\-\u2013\u2014]", v_num)
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
