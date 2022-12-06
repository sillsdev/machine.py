from abc import abstractmethod
from io import TextIOWrapper
from typing import Generator, Iterable, List, Optional, Sequence

from ..scripture.verse_ref import VerseRef, Versification, are_overlapping_verse_ranges
from ..utils.string_utils import has_sentence_ending
from .corpora_utils import gen, merge_verse_ranges
from .scripture_text import ScriptureText
from .stream_container import StreamContainer
from .text_row import TextRow
from .usfm_parser import parse_usfm
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType


class UsfmTextBase(ScriptureText):
    def __init__(
        self,
        id: str,
        stylesheet: UsfmStylesheet,
        encoding: str,
        versification: Optional[Versification],
        include_markers: bool,
    ) -> None:
        super().__init__(id, versification)

        self._stylesheet = stylesheet
        self._encoding = encoding
        self._include_markers = include_markers

    @abstractmethod
    def _create_stream_container(self) -> StreamContainer:
        ...

    def _get_rows(self) -> Generator[TextRow, None, None]:
        usfm = self._read_usfm()
        row_collector = _TextRowCollector(self)
        parse_usfm(usfm, row_collector, self._stylesheet, self.versification, preserve_whitespace=self._include_markers)
        return gen(row_collector.rows)

    def _read_usfm(self) -> str:
        with self._create_stream_container() as stream_container, TextIOWrapper(
            stream_container.open_stream(), encoding=self._encoding, errors="replace"
        ) as reader:
            return reader.read()


class _TextRowCollector(UsfmParserHandler):
    def __init__(self, text: UsfmTextBase) -> None:
        self._text = text
        self._rows: List[TextRow] = []
        self._verse_text = ""
        self._next_para_tokens: List[UsfmToken] = []
        self._verse_ref: Optional[VerseRef] = None
        self._sentence_start: bool = False
        self._next_para_text_started = False

    @property
    def rows(self) -> Iterable[TextRow]:
        return self._rows

    def chapter(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        self._verse_completed(next_sentence_start=True)
        self._verse_ref = None

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        if self._verse_ref is None:
            self._verse_ref = state.verse_ref.copy()
        elif state.verse_ref.exact_equals(self._verse_ref):
            self._verse_completed()

            # ignore duplicate verse
            self._verse_ref = None
        elif are_overlapping_verse_ranges(number, self._verse_ref.verse):
            # merge overlapping verse ranges in to one range
            self._verse_ref.verse = merge_verse_ranges(number, self._verse_ref.verse)
        else:
            self._verse_completed()
            self._verse_ref = state.verse_ref.copy()
        self._next_para_text_started = True
        self._next_para_tokens.clear()

    def start_para(
        self, state: UsfmParserState, marker: str, unknown: bool, attributes: Optional[Sequence[UsfmAttribute]]
    ) -> None:
        self._handle_para(state)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        self._handle_para(state)

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        if self._verse_ref is None:
            return

        if self._text._include_markers:
            self._output_marker(state)
        else:
            if len(self._verse_text) > 0 and not self._verse_text[-1].isspace():
                self._verse_text += " "

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        self._output_marker(state)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self._output_marker(state)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        assert state.prev_token is not None
        if self._text._include_markers and attributes is not None and state.prev_token.type == UsfmTokenType.ATTRIBUTE:
            self._verse_text += str(state.prev_token)

        if closed:
            self._output_marker(state)
        if not self._text._include_markers and marker == "rq":
            self._verse_text = self._verse_text.rstrip()

    def start_note(self, state: UsfmParserState, marker: str, caller: str, category: Optional[str]) -> None:
        self._output_marker(state)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed:
            self._output_marker(state)

    def text(self, state: UsfmParserState, text: str) -> None:
        if self._verse_ref is None or not state.is_verse_para:
            return

        if self._text._include_markers:
            text = text.rstrip("\r\n")
            if len(text) > 0:
                if not text.isspace():
                    for token in self._next_para_tokens:
                        self._verse_text += str(token)
                    self._next_para_tokens.clear()
                    self._next_para_text_started = True
                self._verse_text += text
        elif state.is_verse_text and len(text) > 0:
            if (
                state.prev_token is not None
                and state.prev_token.type == UsfmTokenType.END
                and (self._verse_text == "" or self._verse_text[-1].isspace())
            ):
                text = text.lstrip()
            self._verse_text += text

    def end_usfm(self, state: UsfmParserState) -> None:
        self._verse_completed()

    def _output_marker(self, state: UsfmParserState) -> None:
        if self._verse_ref is None or not self._text._include_markers:
            return

        assert state.token is not None

        if self._next_para_text_started:
            self._verse_text += str(state.token)
        else:
            self._next_para_tokens.append(state.token)

    def _verse_completed(self, next_sentence_start: Optional[bool] = None) -> None:
        if self._verse_ref is None:
            return

        self._rows.extend(self._text._create_rows(self._verse_ref, self._verse_text, self._sentence_start))
        self._sentence_start = (
            has_sentence_ending(self._verse_text) if next_sentence_start is None else next_sentence_start
        )
        self._verse_text = ""

    def _handle_para(self, state: UsfmParserState) -> None:
        if self._verse_ref is None:
            return

        assert state.token is not None

        if state.is_verse_para:
            if len(self._verse_text) > 0 and not self._verse_text[-1].isspace():
                self._verse_text += " "
            self._next_para_tokens.append(state.token)
            self._next_para_text_started = False
