from typing import List, Optional, Tuple, Union

from ..scripture.verse_ref import VerseRef
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer


class UsfmVerseTextUpdater(UsfmParserHandler):
    def __init__(
        self,
        rows: Optional[List[Tuple[List[VerseRef], str]]] = None,
        id_text: Optional[str] = None,
        strip_all_text: Optional[bool] = False,
    ) -> None:
        self._rows = rows or []
        self._tokens: List[UsfmToken] = []
        self._id_text = id_text
        self._strip_all_text = strip_all_text
        self._row_index: int = 0
        self._token_index: int = 0
        self._replace_text: bool = False

    @property
    def tokens(self) -> List[UsfmToken]:
        return self._tokens

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._collect_tokens(state)
        if self._id_text is not None:
            self._tokens.append(UsfmToken(UsfmTokenType.TEXT, text=self._id_text + " "))
            self._replace_text = True

    def end_book(self, state: UsfmParserState, marker: str) -> None:
        self._replace_text = False

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[List[UsfmAttribute]],
    ) -> None:
        if not state.is_verse_para:
            self._replace_text = False
        self._collect_tokens(state)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        self._collect_tokens(state)

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        self._collect_tokens(state)

    def end_cell(self, state: UsfmParserState, marker: str) -> None:
        self._collect_tokens(state)

    def start_sidebar(self, state: UsfmParserState, marker: str, category: str) -> None:
        self._replace_text = False
        self._collect_tokens(state)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self._replace_text = False
        if closed:
            self._collect_tokens(state)

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._replace_text = False
        self._collect_tokens(state)

    def milestone(
        self,
        state: UsfmParserState,
        marker: str,
        start_milestone: bool,
        attributes: List[UsfmAttribute],
    ) -> None:
        self._collect_tokens(state)

    def verse(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._replace_text = False
        self._collect_tokens(state)

        while self._row_index < len(self._rows):
            verse_refs, text = self._rows[self._row_index]
            stop = False
            for verse_ref in verse_refs:
                compare = verse_ref.compare_to(state.verse_ref, compare_segments=False)
                if compare == 0:
                    self._tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text + " "))
                    self._replace_text = True
                    break
                else:
                    if any(v == verse_ref for v in state.verse_ref.all_verses()):
                        self._tokens.append(UsfmToken(UsfmTokenType.TEXT, text=text + " "))
                        self._replace_text = True
                        break
                    if compare > 0:
                        stop = True
                        break
            if stop:
                break
            else:
                self._row_index += 1

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: List[UsfmAttribute],
    ) -> None:
        if self._strip_all_text or (self._replace_text and state.is_verse_para):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def end_char(
        self,
        state: UsfmParserState,
        marker: str,
        attributes: List[UsfmAttribute],
        closed: bool,
    ) -> None:
        if closed and (self._strip_all_text or (self._replace_text and state.is_verse_para)):
            self._skip_tokens(state)

    def start_note(
        self,
        state: UsfmParserState,
        marker: str,
        caller: str,
        category: str,
    ) -> None:
        if self._strip_all_text or (self._replace_text and state.is_verse_para):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed and (self._strip_all_text or (self._replace_text and state.is_verse_para)):
            self._skip_tokens(state)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        if self._strip_all_text or (self._replace_text and state.is_verse_para):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def text(self, state: UsfmParserState, text: str) -> None:
        if self._strip_all_text or (
            self._replace_text and (state.is_verse_para or (state.para_tag and state.para_tag.marker == "id"))
        ):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def opt_break(self, state: UsfmParserState) -> None:
        if self._strip_all_text or (self._replace_text and state.is_verse_para):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def unmatched(self, state: UsfmParserState, marker: str) -> None:
        if self._strip_all_text or (self._replace_text and state.is_verse_para):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def get_usfm(self, stylesheet: Union[str, UsfmStylesheet] = "usfm.sty") -> str:
        if isinstance(stylesheet, str):
            stylesheet = UsfmStylesheet(stylesheet)
        tokenizer = UsfmTokenizer(stylesheet)
        return tokenizer.detokenize(self._tokens)

    def _collect_tokens(self, state: UsfmParserState) -> None:
        while self._token_index <= state.index + state.special_token_count:
            self._tokens.append(state.tokens[self._token_index])
            self._token_index += 1

    def _skip_tokens(self, state: UsfmParserState) -> None:
        self._token_index = state.index + 1 + state.special_token_count
