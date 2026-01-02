from enum import Enum, auto
from typing import List, Optional

from machine.scripture import canon

from ..scripture.verse_ref import ValidStatus, VerseRef, Versification
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmParserState


class UsfmVersificationErrorType(Enum):
    MISSING_CHAPTER = auto()
    MISSING_VERSE = auto()
    EXTRA_VERSE = auto()
    INVALID_VERSE_RANGE = auto()
    MISSING_VERSE_SEGMENT = auto()
    EXTRA_VERSE_SEGMENT = auto()


class UsfmVersificationError:
    def __init__(
        self,
        book_num: int,
        expected_chapter: int,
        expected_verse: int,
        actual_chapter: int,
        actual_verse: int,
        verse_ref: Optional[VerseRef] = None,
    ):
        self._book_num = book_num
        self._expected_chapter = expected_chapter
        self._expected_verse = expected_verse
        self._actual_chapter = actual_chapter
        self._actual_verse = actual_verse
        self._verse_ref = verse_ref
        self._type: UsfmVersificationErrorType

    @property
    def type(self) -> UsfmVersificationErrorType:
        return self._type

    def check_error(self) -> bool:
        """Returns true if there is an error"""
        if self._expected_chapter > self._actual_chapter and self._expected_verse != 0:
            self._type = UsfmVersificationErrorType.MISSING_CHAPTER
            return True
        if self._expected_verse > self._actual_verse and self._expected_chapter == self._actual_chapter:
            self._type = UsfmVersificationErrorType.MISSING_VERSE
            return True
        if self._verse_ref is not None:
            if not self._verse_ref.segment() and self._verse_ref.has_segments_defined:
                self._type = UsfmVersificationErrorType.MISSING_VERSE_SEGMENT
                return True
            if self._verse_ref.segment() and not self._verse_ref.has_segments_defined:
                self._type = UsfmVersificationErrorType.EXTRA_VERSE_SEGMENT
                return True
            if not self._verse_ref.is_valid:
                self._type = UsfmVersificationError.map(self._verse_ref.valid_status)
                return True
        return False

    @staticmethod
    def map(valid_status: ValidStatus) -> UsfmVersificationErrorType:
        if valid_status == ValidStatus.OUT_OF_RANGE:
            return UsfmVersificationErrorType.EXTRA_VERSE
        if valid_status == ValidStatus.VERSE_REPEATED or valid_status == ValidStatus.VERSE_OUT_OF_ORDER:
            return UsfmVersificationErrorType.INVALID_VERSE_RANGE
        raise ValueError(
            f"{ValidStatus.__name__} {valid_status} does not map to any {UsfmVersificationErrorType.__name__}"
        )

    @property
    def expected_verse_ref(self) -> str:
        if (
            default_verse_ref := VerseRef.try_from_string(
                f"{self._book_num} {self._expected_chapter}:{self._expected_verse}"
            )
            is None
        ):
            return self.default_verse(self._expected_chapter, self._expected_verse)
        if self._type == UsfmVersificationErrorType.EXTRA_VERSE:
            return ""
        if self._type == UsfmVersificationErrorType.MISSING_VERSE_SEGMENT:
            if (
                verse_ref_with_segment := VerseRef.try_from_string(
                    f"{self._book_num} {self._expected_chapter}:{self._expected_verse}a"
                )
                is not None
            ):
                return str(verse_ref_with_segment)
        if self._type == UsfmVersificationErrorType.INVALID_VERSE_RANGE and self._verse_ref is not None:
            sorted_all_unique_verses = sorted(set(self._verse_ref.all_verses()))
            first_verse = sorted_all_unique_verses[0]
            last_verse = sorted_all_unique_verses[-1]
            if first_verse == last_verse:
                return str(first_verse)
            elif (
                corrected_verse_range_ref := VerseRef.try_from_string(
                    f"{self._book_num} {self._expected_chapter}:{first_verse}-{last_verse}"
                )
                is not None
            ):
                return str(corrected_verse_range_ref)
        return str(default_verse_ref)

    @property
    def actual_verse_ref(self) -> str:
        if self._verse_ref is not None:
            return str(self._verse_ref)
        if actual_verse_ref := VerseRef.try_from_string(
            f"{self._book_num} {self._actual_chapter}:{self._actual_verse}"
        ):
            return str(actual_verse_ref)
        return self.default_verse(self._actual_chapter, self._actual_verse)

    def default_verse(self, chapter: int, verse: int):
        verse_string = "" if self._actual_verse == -1 else str(verse)
        return f"{canon.book_number_to_id(self._book_num)} {chapter}:{verse_string}"


class UsfmVersificationErrorDetector(UsfmParserHandler):
    def __init__(self, versification: Versification):
        self._versification = versification
        self._current_book = 0
        self._current_chapter = 0
        self._current_verse = VerseRef()
        self._errors: List[UsfmVersificationError] = []

    @property
    def errors(self) -> List[UsfmVersificationError]:
        return self._errors.copy()

    def end_usfm(self, state: UsfmParserState) -> None:
        if self._current_book > 0 and canon.is_canonical(self._current_book):
            versification_error = UsfmVersificationError(
                self._current_book,
                self._versification.get_last_chapter(self._current_book),
                self._versification.get_last_verse(
                    self._current_book, self._versification.get_last_chapter(self._current_book)
                ),
                self._current_chapter,
                list(self._current_verse.all_verses())[-1].verse_num,
            )
            if versification_error.check_error():
                self._errors.append(versification_error)

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._current_book = state.verse_ref.book_num
        self._current_chapter = 0
        self._current_verse = VerseRef()

    def chapter(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        if self._current_book > 0 and canon.is_canonical(self._current_book) and self._current_chapter > 0:
            versification_error = UsfmVersificationError(
                self._current_book,
                self._current_chapter,
                self._versification.get_last_verse(self._current_book, self._current_chapter),
                self._current_chapter,
                list(self._current_verse.all_verses())[-1].verse_num,
            )
            if versification_error.check_error():
                self._errors.append(versification_error)

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        if self._current_book > 0 and canon.is_canonical(self._current_book) and self._current_chapter > 0:
            versification_error = UsfmVersificationError(
                self._current_book,
                self._current_chapter,
                list(self._current_verse.all_verses())[-1].verse_num,
                self._current_chapter,
                list(self._current_verse.all_verses())[-1].verse_num,
            )
            if versification_error.check_error():
                self._errors.append(versification_error)
