from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

from ..string_utils import is_integer, parse_integer
from .canon import LAST_BOOK, book_id_to_number, book_number_to_id

if TYPE_CHECKING:
    from .versification import Versification

_VERSE_RANGE_SEPARATOR = "-"
_VERSE_SEQUENCE_INDICATOR = ","
_CHAPTER_DIGIT_SHIFTER = 1000
_BOOK_DIGIT_SHIFTER = _CHAPTER_DIGIT_SHIFTER * _CHAPTER_DIGIT_SHIFTER
_BCV_MAX_VALUE = _CHAPTER_DIGIT_SHIFTER


def _is_verse_parseable(verse: str) -> bool:
    return (
        len(verse) != 0
        and is_integer(verse[0])
        and verse[-1] != _VERSE_RANGE_SEPARATOR
        and verse[-1] != _VERSE_SEQUENCE_INDICATOR
    )


def _get_verse_num(verse: Optional[str]) -> Tuple[bool, int]:
    if verse is None or verse == "":
        return True, -1

    v_num = 0
    for i in range(len(verse)):
        ch = verse[i]
        if not is_integer(ch):
            if i == 0:
                v_num - 1
            return False, v_num

        v_num = (v_num * 10) + int(ch)
        if v_num > 999:
            v_num = -1
            return False, v_num
    return True, v_num


def get_bbbcccvvv(book_num: int, chapter_num: int, verse_num: int) -> int:
    return (
        (book_num % _BCV_MAX_VALUE) * _BOOK_DIGIT_SHIFTER
        + ((chapter_num % _BCV_MAX_VALUE) * _CHAPTER_DIGIT_SHIFTER if chapter_num >= 0 else 0)
        + (verse_num % _BCV_MAX_VALUE if verse_num >= 0 else 0)
    )


class VerseRef:
    def __init__(
        self,
        book: Union[str, int],
        chapter: Union[str, int],
        verse: Union[str, int],
        versification: Optional["Versification"] = None,
    ) -> None:
        from .versification import VersificationType, get_builtin_versification

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
            get_builtin_versification(VersificationType.ENGLISH) if versification is None else versification
        )

    @classmethod
    def from_string(cls, verse_str: str, versification: Optional["Versification"] = None) -> "VerseRef":
        from .versification import get_builtin_versification

        verse_str = verse_str.replace("\u200f", "")
        if verse_str.find("/") >= 0:
            parts = verse_str.split("/")
            verse_str = parts[0]
            if len(parts) > 1:
                type = parse_integer(parts[1].strip())
                if type is None:
                    raise ValueError("The verse reference is invalid.")
                versification = get_builtin_versification(type)

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
    def has_multiple(self) -> bool:
        return self._verse is not None and (
            self._verse.find(_VERSE_RANGE_SEPARATOR) != -1 or self._verse.find(_VERSE_SEQUENCE_INDICATOR) != -1
        )

    def get_segments(self, default_segments: Optional[List[str]] = None) -> Optional[List[str]]:
        if self.versification is None:
            return default_segments

        segments = self.versification.verse_segments(self.bbbcccvvv)
        if segments is None:
            segments = default_segments
        return segments

    def validated_segment(self, valid_segments: Optional[List[str]] = None) -> str:
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
            if c == _VERSE_RANGE_SEPARATOR or c == _VERSE_SEQUENCE_INDICATOR:
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
            parts = self._verse.split(_VERSE_SEQUENCE_INDICATOR)
            for part in parts:
                pieces = part.split(_VERSE_RANGE_SEPARATOR)
                start_verse = VerseRef(self.book, self.chapter, pieces[0])
                yield start_verse

                if len(pieces) > 1:
                    last_verse = VerseRef(self.book, self.chapter, pieces[1])
                    for verse_num in range(start_verse.verse_num + 1, last_verse.verse_num):
                        yield VerseRef(self.book, self.chapter, str(verse_num))
                    yield last_verse

    def copy(self) -> "VerseRef":
        copy = VerseRef(self.book_num, self.chapter_num, self.verse_num, self.versification)
        copy._verse = self._verse
        return copy

    def copy_from(self, vref: "VerseRef") -> None:
        self.book_num = vref.book_num
        self.chapter_num = vref.chapter_num
        self.verse_num = vref.verse_num
        self._verse = vref._verse
        self.versification = vref.versification

    def change_versification(self, versification: "Versification") -> None:
        if not self.has_multiple:
            versification.change_versification(self)
        else:
            _, result = versification.change_versification_with_ranges(self)
            self.copy_from(result)

    def change_versification_with_ranges(self, versification: "Versification") -> bool:
        result, temp = versification.change_versification_with_ranges(self)
        self.copy_from(temp)
        return result

    def __eq__(self, other: "VerseRef") -> bool:
        return self._compare(other) == 0

    def __lt__(self, other: "VerseRef") -> bool:
        return self._compare(other) < 0

    def __gt__(self, other: "VerseRef") -> bool:
        return self._compare(other) > 0

    def __le__(self, other: "VerseRef") -> bool:
        return self._compare(other) <= 0

    def __ge__(self, other: "VerseRef") -> bool:
        return self._compare(other) >= 0

    def __hash__(self) -> int:
        if self._verse is not None:
            return self.bbbcccvvv ^ hash(self._verse)
        return self.bbbcccvvv

    def __repr__(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def _compare(self, other: "VerseRef") -> int:
        if self.versification is not None and self.versification != other.versification:
            other = other.copy()
            if self.has_multiple:
                other.change_versification_with_ranges(self.versification)
            else:
                other.change_versification(self.versification)

        if self.book_num != other.book_num:
            return self.book_num - other.book_num
        if self.chapter_num != other.chapter_num:
            return self.chapter_num - other.chapter_num

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
