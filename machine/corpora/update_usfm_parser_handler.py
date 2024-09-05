from typing import List, Optional, Sequence, Tuple, Union

from .scripture_ref import ScriptureRef
from .scripture_ref_usfm_parser_handler import ScriptureRefUsfmParserHandler
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer


class UpdateUsfmParserHandler(ScriptureRefUsfmParserHandler):
    def __init__(
        self,
        rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
        id_text: Optional[str] = None,
        strip_all_text: bool = False,
        prefer_existing_text: bool = False,
    ) -> None:
        super().__init__()
        self._rows = rows or []
        self._tokens: List[UsfmToken] = []
        self._new_tokens: List[UsfmToken] = []
        self._id_text = id_text
        self._strip_all_text = strip_all_text
        self._prefer_existing_text = prefer_existing_text
        self._replace_stack: List[bool] = []
        self._row_index: int = 0
        self._token_index: int = 0

    @property
    def tokens(self) -> List[UsfmToken]:
        return self._tokens

    def end_usfm(self, state: UsfmParserState) -> None:
        self._collect_tokens(state)

        super().end_usfm(state)

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._collect_tokens(state)
        start_book_tokens: List[UsfmToken] = []
        if self._id_text is not None:
            start_book_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=self._id_text + " "))
        self._push_new_tokens(start_book_tokens)

        super().start_book(state, marker, code)

    def end_book(self, state: UsfmParserState, marker: str) -> None:
        self._pop_new_tokens()

        super().end_book(state, marker)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self._collect_tokens(state)

        super().start_para(state, marker, unknown, attributes)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        self._collect_tokens(state)

        super().start_row(state, marker)

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        self._collect_tokens(state)

        super().start_cell(state, marker, align, colspan)

    def end_cell(self, state: UsfmParserState, marker: str) -> None:
        self._collect_tokens(state)

        super().end_cell(state, marker)

    def start_sidebar(self, state: UsfmParserState, marker: str, category: str) -> None:
        self._collect_tokens(state)

        super().start_sidebar(state, marker, category)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed:
            self._collect_tokens(state)

        super().end_sidebar(state, marker, closed)

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._collect_tokens(state)

        super().chapter(state, number, marker, alt_number, pub_number)

    def milestone(
        self,
        state: UsfmParserState,
        marker: str,
        start_milestone: bool,
        attributes: Sequence[UsfmAttribute],
    ) -> None:
        self._collect_tokens(state)

        super().milestone(state, marker, start_milestone, attributes)

    def verse(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._collect_tokens(state)

        super().verse(state, number, marker, alt_number, pub_number)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Sequence[UsfmAttribute],
    ) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().start_char(state, marker_without_plus, unknown, attributes)

    def end_char(
        self,
        state: UsfmParserState,
        marker: str,
        attributes: Sequence[UsfmAttribute],
        closed: bool,
    ) -> None:
        if closed and self._replace_with_new_tokens(state):
            self._skip_tokens(state)

        super().end_char(state, marker, attributes, closed)

    def start_note(
        self,
        state: UsfmParserState,
        marker: str,
        caller: str,
        category: str,
    ) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().start_note(state, marker, caller, category)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed and self._replace_with_new_tokens(state):
            self._skip_tokens(state)

        super().end_note(state, marker, closed)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().ref(state, marker, display, target)

    def text(self, state: UsfmParserState, text: str) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().text(state, text)

    def opt_break(self, state: UsfmParserState) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().opt_break(state)

    def unmatched(self, state: UsfmParserState, marker: str) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().unmatched(state, marker)

    def _start_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        row_texts: List[str] = self._advance_rows(scripture_refs)
        self._push_new_tokens([UsfmToken(UsfmTokenType.TEXT, text=t + " ") for t in row_texts])

    def _end_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        self._pop_new_tokens()

    def _start_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        row_texts = self._advance_rows([scripture_ref])
        self._push_new_tokens([UsfmToken(UsfmTokenType.TEXT, text=t + " ") for t in row_texts])

    def _end_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._pop_new_tokens()

    def _start_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        row_texts = self._advance_rows([scripture_ref])
        new_tokens: List[UsfmToken] = []
        if len(row_texts) > 0:
            if state.token is None:
                raise ValueError("Invalid parser state.")
            new_tokens.append(state.token)
            new_tokens.append(UsfmToken(UsfmTokenType.CHARACTER, "ft", None, "ft*"))
            for i, text in enumerate(row_texts):
                if i < len(row_texts) - 1:
                    text += " "
                new_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text))
            new_tokens.append(UsfmToken(UsfmTokenType.END, state.token.end_marker, None, None))
            self._push_new_tokens(new_tokens)
        else:
            self._push_token_as_previous()

    def _end_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._pop_new_tokens()

    def get_usfm(self, stylesheet: Union[str, UsfmStylesheet] = "usfm.sty") -> str:
        if isinstance(stylesheet, str):
            stylesheet = UsfmStylesheet(stylesheet)
        tokenizer = UsfmTokenizer(stylesheet)
        return tokenizer.detokenize(self._tokens)

    def _advance_rows(self, seg_scr_refs: Sequence[ScriptureRef]) -> List[str]:
        row_texts: List[str] = []
        source_index: int = 0
        while self._row_index < len(self._rows) and source_index < len(seg_scr_refs):
            compare: int = 0
            row_scr_refs, text = self._rows[self._row_index]
            for row_scr_ref in row_scr_refs:
                while source_index < len(seg_scr_refs):
                    compare = row_scr_ref.compare_to(seg_scr_refs[source_index], compare_segments=False)
                    if compare > 0:
                        # row is ahead of source, increment source
                        source_index += 1
                    else:
                        break
                if compare == 0:
                    # source and row match
                    # grab the text - both source and row will be incremented in due time...
                    row_texts.append(text)
                    break
            if compare <= 0:
                # source is ahead of row, increment row
                self._row_index += 1
        return row_texts

    def _collect_tokens(self, state: UsfmParserState) -> None:
        self._tokens.extend(self._new_tokens)
        self._new_tokens.clear()
        while self._token_index <= state.index + state.special_token_count:
            self._tokens.append(state.tokens[self._token_index])
            self._token_index += 1

    def _skip_tokens(self, state: UsfmParserState) -> None:
        self._token_index = state.index + 1 + state.special_token_count

    def _replace_with_new_tokens(self, state: UsfmParserState) -> bool:
        new_text: bool = len(self._replace_stack) > 0 and self._replace_stack[-1]
        token_end: int = state.index + state.special_token_count
        existing_text: bool = False
        for index in range(self._token_index, token_end + 1):
            if state.tokens[index].type == UsfmTokenType.TEXT and state.tokens[index].text:
                existing_text = True
                break
        use_new_tokens: bool = (
            self._strip_all_text or (new_text and not existing_text) or (new_text and not self._prefer_existing_text)
        )
        if use_new_tokens:
            self._tokens.extend(self._new_tokens)
        self._new_tokens.clear()
        return use_new_tokens

    def _push_new_tokens(self, tokens: List[UsfmToken]) -> None:
        self._replace_stack.append(any(tokens))
        self._new_tokens.extend(tokens)

    def _push_token_as_previous(self) -> None:
        self._replace_stack.append(self._replace_stack[-1])

    def _pop_new_tokens(self) -> None:
        self._replace_stack.pop()
