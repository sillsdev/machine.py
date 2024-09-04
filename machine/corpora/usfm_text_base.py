from abc import abstractmethod
from io import TextIOWrapper
from typing import Generator, Iterable, List, Optional, Sequence

from ..scripture.verse_ref import Versification
from ..utils.string_utils import has_sentence_ending
from .corpora_utils import gen
from .scripture_ref import ScriptureRef
from .scripture_ref_usfm_parser_handler import ScriptureRefUsfmParserHandler, ScriptureTextType
from .scripture_text import ScriptureText
from .stream_container import StreamContainer
from .text_row import TextRow
from .usfm_parser import UsfmParser
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer


class UsfmTextBase(ScriptureText):
    def __init__(
        self,
        id: str,
        stylesheet: UsfmStylesheet,
        encoding: str,
        versification: Optional[Versification],
        include_markers: bool,
        include_all_text: bool,
        project: Optional[str] = None,
    ) -> None:
        super().__init__(id, versification)

        self._stylesheet = stylesheet
        self._encoding = encoding
        self._include_markers = include_markers
        self._include_all_text = include_all_text
        self.project = project

    @abstractmethod
    def _create_stream_container(self) -> StreamContainer: ...

    def _get_rows(self) -> Generator[TextRow, None, None]:
        usfm = self._read_usfm()
        row_collector = _TextRowCollector(self)

        tokenizer = UsfmTokenizer(self._stylesheet)
        try:
            tokens = tokenizer.tokenize(usfm, self._include_markers)
        except Exception as e:
            error_message = (
                f"An error occurred while tokenizing the text '{self.id}'"
                f"{f' in project {self.project}' if self.project else ''}"
                f". Error: '{e}'"
            )
            raise RuntimeError(error_message) from e

        parser = UsfmParser(tokens, row_collector, self._stylesheet, self._versification, self._include_markers)
        try:
            parser.process_tokens()
        except Exception as e:
            error_message = (
                f"An error occurred while parsing the text '{self.id}'"
                f"{f' in project {self.project}' if self.project else ''}"
                f". Verse: {parser.state.verse_ref}, line: {parser.state.line_number}, "
                f"character: {parser.state.line_number}, error: '{e}'"
            )
            raise RuntimeError(error_message) from e
        return gen(row_collector.rows)

    def _read_usfm(self) -> str:
        with self._create_stream_container() as stream_container, TextIOWrapper(
            stream_container.open_stream(), encoding=self._encoding, errors="replace"
        ) as reader:
            return reader.read()


class _TextRowCollector(ScriptureRefUsfmParserHandler):
    def __init__(self, text: UsfmTextBase) -> None:
        super().__init__()

        self._text = text
        self._rows: List[TextRow] = []
        self._next_para_tokens: List[UsfmToken] = []
        self._row_texts_stack: List[str] = []
        self._sentence_start: bool = False
        self._next_para_text_started = False

    @property
    def rows(self) -> Iterable[TextRow]:
        return self._rows

    def verse(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        super().verse(state, number, marker, alt_number, pub_number)
        self._next_para_text_started = True
        self._next_para_tokens.clear()

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        super().start_para(state, marker, unknown, attributes)
        self._handle_para(state)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        super().start_row(state, marker)
        self._handle_para(state)

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        super().start_cell(state, marker, align, colspan)

        if self._text._include_markers:
            self._output_marker(state)
        elif self._current_text_type == ScriptureTextType.VERSE:
            if len(self._row_texts_stack) == 0:
                return
            verse_text: str = self._row_texts_stack[-1]
            if len(verse_text) > 0 and not verse_text[-1].isspace():
                self._row_texts_stack[-1] += " "

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        super().ref(state, marker, display, target)
        self._output_marker(state)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        super().start_char(state, marker_without_plus, unknown, attributes)
        self._output_marker(state)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        assert state.prev_token is not None
        super().end_char(state, marker, attributes, closed)

        if not self._row_texts_stack:
            return

        if self._text._include_markers and attributes is not None and state.prev_token.type == UsfmTokenType.ATTRIBUTE:
            self._row_texts_stack[-1] += str(state.prev_token)

        if closed:
            self._output_marker(state)
        if not self._text._include_markers and marker == "rq":
            self._row_texts_stack[-1] = self._row_texts_stack[-1].rstrip()

    def start_note(self, state: UsfmParserState, marker: str, caller: str, category: Optional[str]) -> None:
        super().start_note(state, marker, caller, category)
        self._output_marker(state)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        super().end_note(state, marker, closed)
        if closed:
            self._output_marker(state)

    def opt_break(self, state: UsfmParserState) -> None:
        super().opt_break(state)
        if self._text._include_markers:
            self._row_texts_stack[-1] += "//"
        elif self._current_text_type != ScriptureTextType.VERSE or state.is_verse_text:
            self._row_texts_stack[-1] = self._row_texts_stack[-1].rstrip()

    def text(self, state: UsfmParserState, text: str) -> None:
        super().text(state, text)

        if len(self._row_texts_stack) == 0:
            return

        row_text = self._row_texts_stack[-1]
        if self._text._include_markers:
            text = text.rstrip("\r\n")
            if len(text) > 0:
                if not text.isspace():
                    for token in self._next_para_tokens:
                        row_text += str(token)
                    self._next_para_tokens.clear()
                    self._next_para_text_started = True
                if len(row_text) == 0 or row_text[-1].isspace():
                    text = text.lstrip()
                row_text += text
        elif len(text) > 0 and (self._current_text_type != ScriptureTextType.VERSE or state.is_verse_text):
            if (
                state.prev_token is not None
                and state.prev_token.type == UsfmTokenType.END
                and (len(row_text) == 0 or row_text[-1].isspace())
            ):
                text = text.lstrip()
            row_text += text
        self._row_texts_stack[-1] = row_text

    def _start_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        self._row_texts_stack.append("")

    def _end_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        text = self._row_texts_stack.pop()
        self._rows.extend(self._text._create_scripture_rows(scripture_refs, text, self._sentence_start))
        self._sentence_start = (state.token and state.token.marker == "c") or has_sentence_ending(text)

    def _start_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._row_texts_stack.append("")

    def _end_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        text = self._row_texts_stack.pop()
        if self._text._include_all_text:
            self._rows.append(self._text._create_scripture_row(scripture_ref, text, self._sentence_start))

    def _start_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        if self._text._include_markers:
            return
        self._row_texts_stack.append("")

    def _end_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        if self._text._include_markers:
            return
        text = self._row_texts_stack.pop()
        if self._text._include_all_text:
            self._rows.append(self._text._create_scripture_row(scripture_ref, text, self._sentence_start))

    def _output_marker(self, state: UsfmParserState) -> None:
        if not self._text._include_markers or len(self._row_texts_stack) == 0:
            return

        assert state.token is not None

        if self._next_para_text_started:
            self._row_texts_stack[-1] += str(state.token)
        else:
            self._next_para_tokens.append(state.token)

    def _handle_para(self, state: UsfmParserState) -> None:
        if len(self._row_texts_stack) == 0:
            return

        assert state.token is not None

        for i, row_text in enumerate(self._row_texts_stack):
            if len(row_text) > 0 and not row_text[-1].isspace():
                self._row_texts_stack[i] += " "
        if self._current_text_type == ScriptureTextType.VERSE:
            self._next_para_tokens.append(state.token)
            self._next_para_text_started = False
        if not state.is_verse_para:
            self._sentence_start = True
