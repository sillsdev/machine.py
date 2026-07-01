from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Sequence, Set

from machine.scripture import canon

from ..scripture.constants import ORIGINAL_VERSIFICATION
from ..scripture.verse_ref import ValidStatus, VerseRef
from .paratext_project_settings import ParatextProjectSettings
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmParserState


class UsfmVersificationDiagnosticType(Enum):
    MISSING = auto()  # Missing content
    EXTRA = auto()  # Extra content
    INVALID = auto()  # Invalid verse or chapter reference
    INCORRECT_VERSE_SEGMENT = auto()  # Verse segment in vrs but not in USFM or segment in USFM but not in vrs
    UNSUPPORTED_VERSE_RANGE = auto()  # Verse range that will cross chapter boundaries when mapped to ScrVers.Original


class UsfmVersificationDiagnostic:
    def __init__(
        self,
        type: UsfmVersificationDiagnosticType,
        references: List[VerseRef],
        filename: Optional[str],
        line_numbers: List[int],
    ) -> None:
        self.type = type
        # Expected verses for MISSING, actual verses for EXTRA and INVALID
        self.references = references
        self.filename = filename
        self.line_numbers = line_numbers

    @property
    def num_affected_verses(self) -> int:
        return sum(len(list(vr.all_verses())) for vr in self.references)

    def extend(self, verse_reference: VerseRef, line_number: Optional[int] = None) -> None:
        self._extend_reference(verse_reference)
        if line_number is not None:
            self.line_numbers.append(line_number)

    def _extend_reference(self, verse_reference: VerseRef) -> None:
        if self.references:  # Combine contiguous references
            last_reference = self.references[-1]
            if (
                verse_reference.book == last_reference.book
                and verse_reference.chapter_num == last_reference.chapter_num
            ):
                last_verse_num = list(last_reference.all_verses())[-1].verse_num
                next_verse_num = list(verse_reference.all_verses())[0].verse_num
                first_verse_num = list(last_reference.all_verses())[0].verse_num
                final_verse_num = list(verse_reference.all_verses())[-1].verse_num
                if (
                    next_verse_num == last_verse_num + 1
                    and (
                        updated_verse_reference := VerseRef.try_from_string(
                            f"{verse_reference.book} {verse_reference.chapter_num}:{first_verse_num}-{final_verse_num}"
                        )
                    )
                    is not None
                ):
                    self.references[-1] = updated_verse_reference
                    return

            self.references.append(verse_reference)
        else:
            self.references.append(verse_reference)


class UsfmVersificationAnalysis:
    def __init__(
        self,
        total_num_affected_verses: int,
        total_num_encountered_verses: int,
        diagnostics: Sequence[UsfmVersificationDiagnostic],
        project_settings: ParatextProjectSettings,
    ) -> None:
        self.total_num_affected_verses = total_num_affected_verses
        self.total_num_encountered_verses = total_num_encountered_verses
        self.diagnostics = diagnostics
        self.project_settings = project_settings


class UsfmVersificationAnalyzerHandler(UsfmParserHandler):
    def __init__(
        self, settings: ParatextProjectSettings, only_chapters: Optional[Dict[int, Optional[Set[int]]]] = None
    ) -> None:
        self._settings = settings
        self._only_chapters = only_chapters
        self._expected_verses: Iterator[VerseRef] = iter(settings.versification.all_included_verses(only_chapters))
        self._cursor: Optional[VerseRef] = None
        self._has_more = self._move_next()
        self._next_expected_verse = VerseRef()
        self._prev_encountered_verse_ref = VerseRef(1, 1, 0)
        self._diagnostics: List[UsfmVersificationDiagnostic] = []
        self._filename: Optional[str] = None
        self._last_verse_in_error = False
        self._last_verse_was_extra = False
        self._last_verse_was_invalid = False
        self._total_verses_analyzed = 0
        self._last_line_number = 1

    def _move_next(self) -> bool:
        try:
            self._cursor = next(self._expected_verses)
            return True
        except StopIteration:
            self._cursor = None
            return False

    def _get_next_expected_verse(self) -> None:
        self._next_expected_verse = self._cursor if self._cursor is not None else VerseRef()
        self._has_more = self._move_next()

    @property
    def _current_error(self) -> UsfmVersificationDiagnostic:
        return self._diagnostics[-1]

    def get_analysis(self) -> UsfmVersificationAnalysis:
        while self._has_more:
            if not self._last_verse_was_invalid:
                self._get_next_expected_verse()
            self._handle_missing_verse()
            self._last_verse_was_invalid = False
        return UsfmVersificationAnalysis(
            total_num_affected_verses=sum(d.num_affected_verses for d in self._diagnostics),
            total_num_encountered_verses=self._total_verses_analyzed,
            diagnostics=self._diagnostics,
            project_settings=self._settings,
        )

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._filename = self._settings.get_book_file_name(state.verse_ref.book)

    def chapter(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        verse_ref = state.verse_ref.copy()
        if not canon.is_canonical(verse_ref.book):
            return
        verse_ref.chapter = number
        if verse_ref.chapter_num == -1:
            self._diagnostics.append(
                UsfmVersificationDiagnostic(
                    type=UsfmVersificationDiagnosticType.INVALID,
                    references=[verse_ref],
                    filename=self._filename,
                    line_numbers=[state.line_number],
                )
            )
            self._last_verse_in_error = True

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        current_verses = state.verse_ref.copy()
        if self._only_chapters is not None:
            if current_verses.book_num not in self._only_chapters:
                return
            chapters_filter = self._only_chapters[current_verses.book_num]
            if chapters_filter is not None and current_verses.chapter_num not in chapters_filter:
                return

        verse_ref = current_verses.copy()
        if not canon.is_canonical(verse_ref.book):
            return
        verse_ref.verse = number
        invalid_verse_num = verse_ref.verse_num == -1
        bad_verse_range = verse_ref.valid_status in (ValidStatus.VERSE_OUT_OF_ORDER, ValidStatus.VERSE_REPEATED)
        if invalid_verse_num or bad_verse_range:
            self._handle_invalid_verse(state, verse_ref)
            self._last_verse_was_invalid = True
        else:
            self._last_verse_was_invalid = False

        segment_mismatch = (not current_verses.segment()) == current_verses.has_segments_defined
        if segment_mismatch:
            self._handle_incorrect_verse_segment(state, verse_ref)

        if current_verses.has_multiple:
            copy = current_verses.copy()
            has_cross_chapter_verse_range = not copy.change_versification(ORIGINAL_VERSIFICATION)
            if has_cross_chapter_verse_range:
                self._diagnostics.append(
                    UsfmVersificationDiagnostic(
                        type=UsfmVersificationDiagnosticType.UNSUPPORTED_VERSE_RANGE,
                        references=[current_verses],
                        filename=self._filename,
                        line_numbers=[state.line_number],
                    )
                )

        for current_verse in sorted(current_verses.all_verses()):
            # Properly handle verse segments
            if self._prev_encountered_verse_ref.compare_to(current_verse, compare_segments=False) < 0:
                self._total_verses_analyzed += 1
                if not self._last_verse_was_extra and self._has_more:
                    self._get_next_expected_verse()
            if self._next_expected_verse.is_default:
                continue
            compare = self._next_expected_verse.compare_to(current_verse, compare_segments=False)
            if compare < 0 and self._has_more:
                self._handle_missing_verse()
                self._get_next_expected_verse()
                while (
                    self._has_more and self._next_expected_verse.compare_to(current_verse, compare_segments=False) < 0
                ):
                    self._current_error.extend(self._next_expected_verse)
                    self._get_next_expected_verse()
            elif (compare > 0 and not self._last_verse_was_invalid) or (compare < 0 and not self._has_more):
                # We want Invalid and Extra to be mutually exclusive to avoid duplicate errors for every
                # Invalid/Extra verse
                if not self._has_more and self._next_expected_verse.compare_to(self._prev_encountered_verse_ref) > 0:
                    self._diagnostics.append(
                        UsfmVersificationDiagnostic(
                            type=UsfmVersificationDiagnosticType.MISSING,
                            references=[self._next_expected_verse],
                            filename=self._filename,
                            line_numbers=[self._last_line_number],
                        )
                    )

                self._handle_extra_verse(state.line_number, current_verse)
            else:
                self._last_verse_in_error = False
            if compare <= 0:
                self._last_verse_was_extra = False

            self._prev_encountered_verse_ref = current_verse

        self._last_line_number = state.line_number

    def _handle_invalid_verse(self, state: UsfmParserState, verse_ref: VerseRef) -> None:
        self._diagnostics.append(
            UsfmVersificationDiagnostic(
                type=UsfmVersificationDiagnosticType.INVALID,
                references=[verse_ref],
                filename=self._filename,
                line_numbers=[state.line_number],
            )
        )
        self._last_verse_in_error = True

    def _handle_incorrect_verse_segment(self, state: UsfmParserState, verse_ref: VerseRef) -> None:
        self._diagnostics.append(
            UsfmVersificationDiagnostic(
                type=UsfmVersificationDiagnosticType.INCORRECT_VERSE_SEGMENT,
                references=[verse_ref],
                filename=self._filename,
                line_numbers=[state.line_number],
            )
        )
        self._last_verse_in_error = True

    def _handle_extra_verse(self, line_number: int, current_verse: VerseRef) -> None:
        if not self._last_verse_in_error or (
            self._last_verse_in_error and self._current_error.type != UsfmVersificationDiagnosticType.EXTRA
        ):
            self._diagnostics.append(
                UsfmVersificationDiagnostic(
                    type=UsfmVersificationDiagnosticType.EXTRA,
                    references=[current_verse],
                    filename=self._filename,
                    line_numbers=[line_number],
                )
            )
            self._last_verse_in_error = True
        else:
            self._current_error.extend(current_verse, line_number)
        self._last_verse_was_extra = True

    def _handle_missing_verse(self) -> None:
        if not self._last_verse_in_error or (
            self._last_verse_in_error and self._current_error.type != UsfmVersificationDiagnosticType.MISSING
        ):
            self._diagnostics.append(
                UsfmVersificationDiagnostic(
                    type=UsfmVersificationDiagnosticType.MISSING,
                    references=[self._next_expected_verse],
                    filename=self._filename,
                    line_numbers=[self._last_line_number],
                )
            )
            self._last_verse_in_error = True
        else:
            self._current_error.extend(self._next_expected_verse)
