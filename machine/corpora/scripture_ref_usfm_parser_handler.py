from abc import ABC
from enum import Enum, auto
from typing import List, Optional, Sequence

from ..scripture.verse_ref import VerseRef, are_overlapping_verse_ranges
from .corpora_utils import merge_verse_ranges
from .scripture_element import ScriptureElement
from .scripture_ref import ScriptureRef
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmParserState
from .usfm_token import UsfmAttribute


class ScriptureTextType(Enum):
    NONE = auto()
    NONVERSE = auto()
    VERSE = auto()
    EMBED = auto()
    NOTE_TEXT = auto()


class ScriptureRefUsfmParserHandler(UsfmParserHandler, ABC):
    def __init__(self) -> None:
        self._cur_verse_ref: VerseRef = VerseRef()
        self._cur_elements_stack: List[ScriptureElement] = []
        self._cur_text_type_stack: List[ScriptureTextType] = []
        self._duplicate_verse: bool = False
        self._in_embed: bool = False
        self._in_note_text: bool = False

    @property
    def _current_text_type(self) -> ScriptureTextType:
        return ScriptureTextType.NONE if len(self._cur_text_type_stack) == 0 else self._cur_text_type_stack[-1]

    def get_in_note_text(self) -> bool:
        return self._in_note_text

    def end_usfm(self, state: UsfmParserState) -> None:
        self._end_verse_text_wrapper(state)

    def chapter(self, state: UsfmParserState, number: str, marker: str, alt_number: str, pub_number: str) -> None:
        self._end_verse_text_wrapper(state)
        self._update_verse_ref(state.verse_ref, marker)

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        if state.verse_ref == self._cur_verse_ref and not self._duplicate_verse:
            self._end_verse_text(state, self._create_verse_refs())
            # ignore duplicate verses
            self._duplicate_verse = True
        elif are_overlapping_verse_ranges(verse1=number, verse2=self._cur_verse_ref.verse):
            # merge overlapping verse ranges in to one range
            verse_ref: VerseRef = self._cur_verse_ref.copy()
            verse_ref.verse = merge_verse_ranges(number, self._cur_verse_ref.verse)
            self._update_verse_ref(verse_ref, marker)
        else:
            if self._current_text_type == ScriptureTextType.NONVERSE:
                self._end_non_verse_text_wrapper(state)
            elif self._current_text_type == ScriptureTextType.VERSE:
                self._end_verse_text_wrapper(state)
            self._update_verse_ref(state.verse_ref, marker)
            self._start_verse_text_wrapper(state)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: Optional[bool],
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        if self._cur_verse_ref.is_default:
            self._update_verse_ref(state.verse_ref, marker)
        if not state.is_verse_text:
            self._start_parent_element(marker)
            self._start_non_verse_text_wrapper(state)

    def end_para(self, state: UsfmParserState, marker: str) -> None:
        if self._current_text_type == ScriptureTextType.NONVERSE:
            self._end_parent_element()
            self._end_non_verse_text_wrapper(state)
        elif self._current_text_type == ScriptureTextType.NONE:
            # empty verse paragraph
            self._start_parent_element(marker)
            self._start_non_verse_text_wrapper(state)
            self._end_parent_element()
            self._end_non_verse_text_wrapper(state)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        if self._current_text_type == ScriptureTextType.NONVERSE or self._current_text_type == ScriptureTextType.NONE:
            self._start_parent_element(marker)

    def end_row(self, state: UsfmParserState, marker: str) -> None:
        if self._current_text_type == ScriptureTextType.NONVERSE or self._current_text_type == ScriptureTextType.NONE:
            self._end_parent_element()

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        if self._current_text_type == ScriptureTextType.NONVERSE or self._current_text_type == ScriptureTextType.NONE:
            self._start_parent_element(marker)
            self._start_non_verse_text_wrapper(state)

    def end_cell(self, state: UsfmParserState, marker: str) -> None:
        if self._current_text_type == ScriptureTextType.NONVERSE:
            self._end_parent_element()
            self._end_non_verse_text_wrapper(state)

    def start_sidebar(self, state: UsfmParserState, marker: str, category: str) -> None:
        self._start_parent_element(marker)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self._end_parent_element()

    def start_note(self, state: UsfmParserState, marker: str, caller: str, category: Optional[str]) -> None:
        self._in_embed = True
        self.start_embed(state, marker, caller, category)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self.end_note_text(state)
        self.end_embed(state, marker, None, closed)
        self._in_embed = False

    def start_embed(self, state: UsfmParserState, marker: str, caller: str, category: Optional[str]) -> None:
        if self._cur_verse_ref.is_default:
            self._update_verse_ref(state.verse_ref, marker)

        if not self._duplicate_verse:
            self._check_convert_verse_para_to_non_verse(state)
            self._next_element(marker)

    def end_embed(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        pass

    def text(self, state: UsfmParserState, text: str) -> None:
        # if we hit text in a verse paragraph and we aren't in a verse, then start a non-verse segment
        if text.strip():
            self._check_convert_verse_para_to_non_verse(state)

    def opt_break(self, state: UsfmParserState) -> None:
        self._check_convert_verse_para_to_non_verse(state)

    def start_char(
        self, state: UsfmParserState, marker: str, unknown: bool, attributes: Optional[Sequence[UsfmAttribute]]
    ) -> None:
        if self._is_embed_part(marker):
            self.end_note_text(state)
        # if we hit a character marker in a verse paragraph and we aren't in a verse, then start a non-verse segment
        self._check_convert_verse_para_to_non_verse(state)

        if self._is_embed_character(marker):
            self._in_embed = True
            self.start_embed(state, marker, "", None)

        if self._is_note_text(marker):
            self.start_note_text(state)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        if self._is_embed_character(marker):
            self.end_embed(state, marker, attributes, closed)
            self._in_embed = False

    def _start_verse_text(self, state: UsfmParserState, scripture_refs: Optional[Sequence[ScriptureRef]]) -> None: ...

    def _end_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None: ...

    def _start_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None: ...

    def _end_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None: ...

    def start_note_text(self, state: UsfmParserState):
        self._in_note_text = True
        self._cur_text_type_stack.append(ScriptureTextType.NOTE_TEXT)
        self._start_note_text(state, self._create_non_verse_ref())

    def _start_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None: ...

    def end_note_text(self, state: UsfmParserState):
        if self._cur_text_type_stack and self._cur_text_type_stack[-1] == ScriptureTextType.NOTE_TEXT:
            self._end_note_text(state, self._create_non_verse_ref())
            self._cur_text_type_stack.pop()
            self._in_note_text = False

    def _end_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None: ...

    def _start_verse_text_wrapper(self, state: UsfmParserState) -> None:
        self._duplicate_verse = False
        self._cur_text_type_stack.append(ScriptureTextType.VERSE)
        self._start_verse_text(state, self._create_verse_refs())

    def _end_verse_text_wrapper(self, state: UsfmParserState) -> None:
        if not self._duplicate_verse and self._cur_verse_ref.verse_num > 0:
            self._end_verse_text(state, self._create_verse_refs())
        if self._cur_verse_ref.verse_num > 0:
            self._cur_text_type_stack.pop()

    def _start_non_verse_text_wrapper(self, state: UsfmParserState) -> None:
        self._cur_text_type_stack.append(ScriptureTextType.NONVERSE)
        self._start_non_verse_text(state, self._create_non_verse_ref())

    def _end_non_verse_text_wrapper(self, state: UsfmParserState) -> None:
        self._end_embed_elements()
        self._end_non_verse_text(state, self._create_non_verse_ref())
        self._cur_text_type_stack.pop()

    def _update_verse_ref(self, verse_ref: VerseRef, marker: str) -> None:
        if not are_overlapping_verse_ranges(verse_ref, self._cur_verse_ref):
            self._cur_elements_stack.clear()
            self._cur_elements_stack.append(ScriptureElement(0, marker))
        self._cur_verse_ref = verse_ref.copy()

    def _next_element(self, marker: str) -> None:
        prev_elem: ScriptureElement = self._cur_elements_stack.pop()
        self._cur_elements_stack.append(ScriptureElement(prev_elem.position + 1, marker))

    def _start_parent_element(self, marker: str) -> None:
        self._next_element(marker)
        self._cur_elements_stack.append(ScriptureElement(0, marker))

    def _end_parent_element(self) -> None:
        self._cur_elements_stack.pop()

    def _end_embed_elements(self) -> None:
        if self._cur_elements_stack and self._is_embed_character(self._cur_elements_stack[-1].name):
            self._cur_elements_stack.pop()

    def _create_verse_refs(self) -> List[ScriptureRef]:
        return (
            [ScriptureRef(v) for v in self._cur_verse_ref.all_verses()]
            if self._cur_verse_ref.has_multiple
            else [ScriptureRef(self._cur_verse_ref)]
        )

    def _create_non_verse_ref(self) -> ScriptureRef:
        verse_ref = (
            list(self._cur_verse_ref.all_verses())[-1] if self._cur_verse_ref.has_multiple else self._cur_verse_ref
        )
        # No need to reverse unlike in Machine, elements are already added in correct order
        path = [e for e in self._cur_elements_stack if e.position > 0]
        return ScriptureRef(verse_ref, path)

    def _check_convert_verse_para_to_non_verse(self, state: UsfmParserState) -> None:
        para_tag = state.para_tag
        if (
            self._current_text_type == ScriptureTextType.NONE
            and para_tag is not None
            and para_tag.marker != "tr"
            and state.is_verse_para
            and self._cur_verse_ref.verse_num == 0
        ):
            self._start_parent_element(para_tag.marker)
            self._start_non_verse_text_wrapper(state)

    def _is_in_embed(self, marker: Optional[str]) -> bool:
        return self._in_embed or self._is_embed_character(marker)

    def _is_note_text(self, marker: Optional[str]) -> bool:
        return marker == "ft"

    def _is_embed_part(self, marker: Optional[str]) -> bool:
        return marker is not None and marker.startswith(("f", "x", "z"))

    def _is_embed_character(self, marker: Optional[str]) -> bool:
        return marker in ("f", "fe", "fig", "fm", "x")
