from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple, Union

from .scripture_ref import ScriptureRef
from .scripture_ref_usfm_parser_handler import ScriptureRefUsfmParserHandler
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer


class UpdateUsfmTextBehavior(Enum):
    PREFER_EXISTING = auto()
    PREFER_NEW = auto()
    STRIP_EXISTING = auto()


class UpdateUsfmMarkerBehavior(Enum):
    PRESERVE = auto()
    STRIP = auto()


class UpdateUsfmParserHandler(ScriptureRefUsfmParserHandler):

    def __init__(
        self,
        rows: Optional[Sequence[Tuple[Sequence[ScriptureRef], str]]] = None,
        id_text: Optional[str] = None,
        text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_EXISTING,
        paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
        preserve_paragraph_styles: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self._rows = rows or []
        self._tokens: List[UsfmToken] = []
        self._new_tokens: List[UsfmToken] = []
        self._new_embed_tokens: List[UsfmToken] = []
        self._id_text = id_text
        if preserve_paragraph_styles is None:
            self._preserve_paragraph_styles = set(["r", "rem"])
        elif isinstance(preserve_paragraph_styles, str):
            self._preserve_paragraph_styles = set([preserve_paragraph_styles])
        else:
            self._preserve_paragraph_styles = set(preserve_paragraph_styles)
        self._text_behavior = text_behavior
        self._paragraph_behavior = paragraph_behavior
        self._embed_behavior = embed_behavior
        self._style_behavior = style_behavior
        self._replace_stack: List[bool] = []
        self._row_index: int = 0
        self._token_index: int = 0
        self._embed_updated: bool = False
        self._embed_row_texts: List[str] = []

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
        if marker in self._preserve_paragraph_styles:
            self._in_preserved_paragraph = True

        if (
            state.is_verse_text
            and (self._has_new_text() or self._text_behavior == UpdateUsfmTextBehavior.STRIP_EXISTING)
            and self._paragraph_behavior == UpdateUsfmMarkerBehavior.STRIP
        ):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().start_para(state, marker, unknown, attributes)

    def end_para(self, state: UsfmParserState, marker: str) -> None:
        super().end_para(state, marker)
        self._in_preserved_paragraph = False

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
        if self._replace_with_new_tokens(state, closed):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().end_char(state, marker, attributes, closed)

    def _start_embed(
        self,
        state: UsfmParserState,
        scripture_ref: ScriptureRef,
    ) -> None:
        self._embed_row_texts = self._advance_rows([scripture_ref])
        self._embed_updated = any(self._embed_row_texts)

        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

    def _end_embed(
        self, state: UsfmParserState, marker: str, attributes: Sequence[UsfmAttribute], closed: bool
    ) -> None:
        if self._replace_with_new_tokens(state, closed):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        self._embed_row_texts.clear()
        self._embed_updated = False

        super()._end_embed(state, marker, attributes, closed)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

        super().ref(state, marker, display, target)

    def text(self, state: UsfmParserState, text: str) -> None:
        super().text(state, text)

        if self._replace_with_new_tokens(state):
            self._skip_tokens(state)
        else:
            self._collect_tokens(state)

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

    def _start_note_text(self, state: UsfmParserState) -> None:
        self._push_new_embed_tokens([UsfmToken(UsfmTokenType.TEXT, text=t + " ") for t in self._embed_row_texts])

    def _end_note_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._embed_row_texts.clear()
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

    def _replace_with_new_tokens(self, state: UsfmParserState, closed: bool = True) -> bool:
        marker: Optional[str] = state.token if state.token is None else state.token.marker
        in_embed: bool = self._is_in_embed(marker)

        in_nested_embed: bool = self._is_in_nested_embed(marker)
        is_style_tag: bool = marker is not None and not self._is_embed_part_style(marker)

        existing_text = any(
            t.type == UsfmTokenType.TEXT and t.text
            for t in state.tokens[self._token_index : state.index + 1 + state.special_token_count]
        )

        use_new_tokens = (
            not self._is_in_preserved_paragraph(marker)
            and (
                self._text_behavior == UpdateUsfmTextBehavior.STRIP_EXISTING
                or (
                    self._has_new_text()
                    and (not existing_text or self._text_behavior != UpdateUsfmTextBehavior.PREFER_EXISTING)
                )
            )
            and (
                not in_embed
                or (
                    self._is_in_note_text()
                    and not in_nested_embed
                    and self._embed_behavior == UpdateUsfmMarkerBehavior.PRESERVE
                )
            )
        )

        if use_new_tokens:
            if in_embed:
                self._add_new_embed_tokens()
            else:
                self._add_new_tokens()

        if existing_text and (
            self._text_behavior == UpdateUsfmTextBehavior.PREFER_EXISTING or self._is_in_preserved_paragraph(marker)
        ):
            if in_embed:
                self._clear_new_embed_tokens()
            else:
                self._clear_new_tokens()

        embed_in_new_verse_text = (
            any(self._replace_stack) or self._text_behavior == UpdateUsfmTextBehavior.STRIP_EXISTING
        ) and in_embed
        if embed_in_new_verse_text or self._embed_updated:
            if self._embed_behavior == UpdateUsfmMarkerBehavior.STRIP:
                self._clear_new_embed_tokens()
                return True
            if not self._is_in_note_text() or in_nested_embed:
                return False

        skip_tokens = use_new_tokens and closed

        if use_new_tokens and is_style_tag:
            skip_tokens = self._style_behavior == UpdateUsfmMarkerBehavior.STRIP

        return skip_tokens

    def _has_new_text(self) -> bool:
        return any(self._replace_stack) and self._replace_stack[-1]

    def _push_new_tokens(self, tokens: List[UsfmToken]) -> None:
        self._replace_stack.append(any(tokens))
        if tokens:
            self._new_tokens.extend(tokens)

    def _add_new_tokens(self) -> None:
        if self._new_tokens:
            self._tokens.extend(self._new_tokens)
        self._new_tokens.clear()

    def _clear_new_tokens(self) -> None:
        self._new_tokens.clear()

    def _push_new_embed_tokens(self, tokens: List[UsfmToken]) -> None:
        self._replace_stack.append(any(tokens))
        if tokens:
            self._new_embed_tokens.extend(tokens)

    def _add_new_embed_tokens(self) -> None:
        if self._new_embed_tokens:
            self._tokens.extend(self._new_embed_tokens)
        self._new_embed_tokens.clear()

    def _clear_new_embed_tokens(self) -> None:
        self._new_embed_tokens.clear()

    def _push_token_as_previous(self) -> None:
        self._replace_stack.append(self._replace_stack[-1])

    def _pop_new_tokens(self) -> None:
        self._replace_stack.pop()

    def _is_in_preserved_paragraph(self, marker: Optional[str]) -> bool:
        return self._in_preserved_paragraph or marker in self._preserve_paragraph_styles
