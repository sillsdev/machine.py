from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ..scripture.verse_ref import VerseRef
from .scripture_embed import is_embed_part_style
from .scripture_ref import ScriptureRef
from .scripture_ref_usfm_parser_handler import ScriptureRefUsfmParserHandler, ScriptureTextType
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_tag import UsfmTextType
from .usfm_token import UsfmAttribute, UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer
from .usfm_update_block import UsfmUpdateBlock
from .usfm_update_block_element import UsfmUpdateBlockElement, UsfmUpdateBlockElementType
from .usfm_update_block_handler import UsfmUpdateBlockHandler


class UpdateUsfmTextBehavior(Enum):
    PREFER_EXISTING = auto()
    PREFER_NEW = auto()
    STRIP_EXISTING = auto()


class UpdateUsfmMarkerBehavior(Enum):
    PRESERVE = auto()
    STRIP = auto()


class UpdateUsfmRow:
    def __init__(self, refs: Sequence[ScriptureRef], text: str, metadata: Optional[dict[str, object]] = None):
        self.refs = refs
        self.text = text
        self.metadata = metadata


class UpdateUsfmParserHandler(ScriptureRefUsfmParserHandler):
    def __init__(
        self,
        rows: Optional[Sequence[UpdateUsfmRow]] = None,
        id_text: Optional[str] = None,
        text_behavior: UpdateUsfmTextBehavior = UpdateUsfmTextBehavior.PREFER_EXISTING,
        paragraph_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        embed_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.PRESERVE,
        style_behavior: UpdateUsfmMarkerBehavior = UpdateUsfmMarkerBehavior.STRIP,
        preserve_paragraph_styles: Optional[Union[Iterable[str], str]] = None,
        update_block_handlers: Optional[Iterable[UsfmUpdateBlockHandler]] = None,
        remarks: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self._rows = rows or []
        self._tokens: List[UsfmToken] = []
        self._updated_text: List[UsfmToken] = []
        self._update_block_stack: list[UsfmUpdateBlock] = []
        self._embed_tokens: List[UsfmToken] = []
        self._id_text = id_text
        if update_block_handlers is None:
            self._update_block_handlers = []
        else:
            self._update_block_handlers = list(update_block_handlers)
        if preserve_paragraph_styles is None:
            self._preserve_paragraph_styles = set(["r", "rem"])
        elif isinstance(preserve_paragraph_styles, str):
            self._preserve_paragraph_styles = set([preserve_paragraph_styles])
        else:
            self._preserve_paragraph_styles = set(preserve_paragraph_styles)
        if remarks is None:
            self._remarks = []
        else:
            self._remarks = list(remarks)
        self._text_behavior = text_behavior
        self._paragraph_behavior = paragraph_behavior
        self._embed_behavior = embed_behavior
        self._style_behavior = style_behavior
        self._replace_stack: List[bool] = []
        self._row_index: int = 0
        self._token_index: int = 0

    @property
    def tokens(self) -> List[UsfmToken]:
        return self._tokens

    def end_usfm(self, state: UsfmParserState) -> None:
        self._collect_updatable_tokens(state)
        super().end_usfm(state)

    def start_book(self, state: UsfmParserState, marker: str, code: str) -> None:
        self._collect_readonly_tokens(state)
        self._update_block_stack.append(UsfmUpdateBlock())
        start_book_tokens: List[UsfmToken] = []
        if self._id_text is not None:
            start_book_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=self._id_text + " "))
        for remark in self._remarks:
            start_book_tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, "rem"))
            start_book_tokens.append(UsfmToken(UsfmTokenType.TEXT, text=remark))
        self._push_updated_text(start_book_tokens)

        super().start_book(state, marker, code)

    def end_book(self, state: UsfmParserState, marker: str) -> None:
        self._use_updated_text()
        self._pop_new_tokens()
        update_block = self._update_block_stack.pop()
        self._tokens.extend(update_block.get_tokens())

        super().end_book(state, marker)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        if state.is_verse_text:
            # Only strip paragraph markers in a verse
            if self._paragraph_behavior == UpdateUsfmMarkerBehavior.PRESERVE:
                self._collect_updatable_tokens(state)
            else:
                self._skip_updatable_tokens(state)
        else:
            self._collect_updatable_tokens(state)

        super().start_para(state, marker, unknown, attributes)

    def start_row(self, state: UsfmParserState, marker: str) -> None:
        self._collect_updatable_tokens(state)

        super().start_row(state, marker)

    def start_cell(self, state: UsfmParserState, marker: str, align: str, colspan: int) -> None:
        self._collect_updatable_tokens(state)

        super().start_cell(state, marker, align, colspan)

    def start_sidebar(self, state: UsfmParserState, marker: str, category: str) -> None:
        self._collect_updatable_tokens(state)

        super().start_sidebar(state, marker, category)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed:
            self._collect_updatable_tokens(state)

        super().end_sidebar(state, marker, closed)

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._use_updated_text()

        super().chapter(state, number, marker, alt_number, pub_number)

        self._collect_readonly_tokens(state)

    def milestone(
        self,
        state: UsfmParserState,
        marker: str,
        start_milestone: bool,
        attributes: Sequence[UsfmAttribute],
    ) -> None:
        self._collect_updatable_tokens(state)

        super().milestone(state, marker, start_milestone, attributes)

    def verse(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: str,
        pub_number: str,
    ) -> None:
        self._use_updated_text()

        # Ensure that a paragraph that contains a verse is not marked for removal
        if len(self._update_block_stack) > 0:
            last_paragraph = self._update_block_stack[-1].get_last_paragraph()
            if last_paragraph is not None:
                last_paragraph.marked_for_removal = False

        super().verse(state, number, marker, alt_number, pub_number)

        self._collect_readonly_tokens(state)

    def start_note(self, state: UsfmParserState, marker: str, caller: str, category: str) -> None:
        super().start_note(state, marker, caller, category)

        self._collect_updatable_tokens(state)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        if closed:
            self._collect_updatable_tokens(state)

        super().end_note(state, marker, closed)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Sequence[UsfmAttribute],
    ) -> None:
        super().start_char(state, marker_without_plus, unknown, attributes)

        if self._current_text_type == ScriptureTextType.EMBED:
            self._collect_updatable_tokens(state)
        else:
            self._replace_with_new_tokens(state)
            if self._style_behavior == UpdateUsfmMarkerBehavior.STRIP:
                self._skip_updatable_tokens(state)
            else:
                self._collect_updatable_tokens(state)

    def end_char(
        self,
        state: UsfmParserState,
        marker: str,
        attributes: Sequence[UsfmAttribute],
        closed: bool,
    ) -> None:
        if closed:
            if self._current_text_type == ScriptureTextType.EMBED:
                self._collect_updatable_tokens(state)
            else:
                self._replace_with_new_tokens(state)
                if self._style_behavior == UpdateUsfmMarkerBehavior.STRIP:
                    self._skip_updatable_tokens(state)
                else:
                    self._collect_updatable_tokens(state)

        super().end_char(state, marker, attributes, closed)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        super().ref(state, marker, display, target)

        if self._replace_with_new_tokens(state):
            self._skip_updatable_tokens(state)
        else:
            self._collect_updatable_tokens(state)

    def text(self, state: UsfmParserState, text: str) -> None:
        super().text(state, text)

        if self._replace_with_new_tokens(state):
            self._skip_updatable_tokens(state)
        else:
            self._collect_updatable_tokens(state)

    def opt_break(self, state: UsfmParserState) -> None:
        super().opt_break(state)

        if self._replace_with_new_tokens(state):
            self._skip_updatable_tokens(state)
        else:
            self._collect_updatable_tokens(state)

    def unmatched(self, state: UsfmParserState, marker: str) -> None:
        super().unmatched(state, marker)

        if self._replace_with_new_tokens(state):
            self._skip_updatable_tokens(state)
        else:
            self._collect_updatable_tokens(state)

    def _start_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        self._collect_updatable_tokens(state)
        self._start_update_block(scripture_refs)

    def _end_verse_text(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        self._end_update_block(state, scripture_refs)

    def _start_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._start_update_block([scripture_ref])

    def _end_non_verse_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._end_update_block(state, [scripture_ref])

    def _end_embed_text(self, state: UsfmParserState, scripture_ref: ScriptureRef) -> None:
        self._update_block_stack[-1].add_embed(
            self._embed_tokens, marked_for_removal=self._embed_behavior == UpdateUsfmMarkerBehavior.STRIP
        )
        self._embed_tokens.clear()

    def get_usfm(self, stylesheet: Union[str, UsfmStylesheet] = "usfm.sty") -> str:
        if isinstance(stylesheet, str):
            stylesheet = UsfmStylesheet(stylesheet)
        tokenizer = UsfmTokenizer(stylesheet)
        return tokenizer.detokenize(self._tokens)

    def _advance_rows(self, seg_scr_refs: Sequence[ScriptureRef]) -> Tuple[List[str], Optional[dict[str, object]]]:
        row_texts: List[str] = []
        row_metadata = None
        source_index: int = 0
        while self._row_index < len(self._rows) and source_index < len(seg_scr_refs):
            compare: int = 0
            row = self._rows[self._row_index]
            row_scr_refs, text, metadata = row.refs, row.text, row.metadata
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
                    row_metadata = metadata
                    break
            if compare <= 0:
                # source is ahead of row, increment row
                self._row_index += 1
        return row_texts, row_metadata

    def _collect_updatable_tokens(self, state: UsfmParserState) -> None:
        self._use_updated_text()
        while self._token_index <= state.index + state.special_token_count:
            token = state.tokens[self._token_index]
            if self._current_text_type == ScriptureTextType.EMBED:
                self._embed_tokens.append(token)
            elif (
                self._current_text_type != ScriptureTextType.NONE
                or (state.para_tag is not None and state.para_tag.marker == "id")
            ) and len(self._update_block_stack) > 0:
                self._update_block_stack[-1].add_token(token)
            else:
                self._tokens.append(token)
            self._token_index += 1

    def _collect_readonly_tokens(self, state: UsfmParserState) -> None:
        while self._token_index <= state.index + state.special_token_count:
            token = state.tokens[self._token_index]
            if len(self._update_block_stack) > 0:
                self._update_block_stack[-1].add_token(token)
            else:
                self._tokens.append(token)
            self._token_index += 1

    def _skip_updatable_tokens(self, state: UsfmParserState) -> None:
        while self._token_index <= state.index + state.special_token_count:
            token = state.tokens[self._token_index]
            if self._current_text_type != ScriptureTextType.NONE or (
                state.para_tag is not None and state.para_tag.marker == "id"
            ):
                if len(self._update_block_stack) > 0:
                    self._update_block_stack[-1].add_token(token, marked_for_removal=True)
            self._token_index += 1
        self._token_index = state.index + 1 + state.special_token_count

    def _replace_with_new_tokens(self, state: UsfmParserState, closed: bool = True) -> bool:
        marker: Optional[str] = state.token if state.token is None else state.token.marker
        in_embed: bool = self._is_in_embed(marker)

        in_nested_embed: bool = self._is_in_nested_embed(marker)
        is_style_tag: bool = marker is not None and not is_embed_part_style(marker)

        existing_text = any(
            t.type == UsfmTokenType.TEXT and t.text
            for t in state.tokens[self._token_index : state.index + 1 + state.special_token_count]
        )

        use_new_tokens = True
        if self._is_in_preserved_paragraph(state):
            use_new_tokens = False
        elif self._text_behavior != UpdateUsfmTextBehavior.STRIP_EXISTING and (
            not self._has_new_text()
            or (existing_text and self._text_behavior == UpdateUsfmTextBehavior.PREFER_EXISTING)
        ):
            use_new_tokens = False

        if use_new_tokens:
            self._use_updated_text()

        clear_new_tokens = existing_text and (
            self._text_behavior == UpdateUsfmTextBehavior.PREFER_EXISTING or self._is_in_preserved_paragraph(state)
        )

        if clear_new_tokens:
            self._clear_updated_text()

        return use_new_tokens

    def _has_new_text(self) -> bool:
        return any(self._replace_stack) and self._replace_stack[-1]

    def _start_update_block(self, scripture_refs: Sequence[ScriptureRef]) -> None:
        row_texts, metadata = self._advance_rows(scripture_refs)
        self._update_block_stack.append(
            UsfmUpdateBlock(scripture_refs, metadata=metadata if metadata is not None else {})
        )
        self._push_updated_text([UsfmToken(UsfmTokenType.TEXT, text=t + " ") for t in row_texts])

    def _end_update_block(self, state: UsfmParserState, scripture_refs: Sequence[ScriptureRef]) -> None:
        self._use_updated_text()
        self._pop_new_tokens()
        update_block = self._update_block_stack.pop()
        update_block.update_refs(scripture_refs)

        # Strip off any non-verse paragraphs that are at the end of the update block
        para_elems: list[UsfmUpdateBlockElement] = []
        while len(update_block.elements) > 0 and _is_nonverse_paragraph(state, update_block.elements[-1]):
            para_elems.append(update_block.pop())

        for handler in self._update_block_handlers:
            update_block = handler.process_block(update_block)
        tokens = update_block.get_tokens()
        for elem in reversed(para_elems):
            tokens.extend(elem.get_tokens())
        if (
            len(self._update_block_stack) > 0
            and self._update_block_stack[-1].elements[-1].type == UsfmUpdateBlockElementType.PARAGRAPH
        ):
            self._update_block_stack[-1].extend_last_element(tokens)
        else:
            self._tokens.extend(tokens)

    def _push_updated_text(self, tokens: List[UsfmToken]) -> None:
        self._replace_stack.append(any(tokens))
        if tokens:
            self._updated_text.extend(tokens)

    def _use_updated_text(self) -> None:
        if self._updated_text:
            self._update_block_stack[-1].add_text(self._updated_text)
        self._updated_text.clear()

    def _clear_updated_text(self) -> None:
        self._updated_text.clear()

    def _pop_new_tokens(self) -> None:
        self._replace_stack.pop()

    def _is_in_preserved_paragraph(self, state: UsfmParserState) -> bool:
        return state.para_tag is not None and state.para_tag.marker in self._preserve_paragraph_styles


def _is_nonverse_paragraph(state: UsfmParserState, element: UsfmUpdateBlockElement) -> bool:
    if element.type != UsfmUpdateBlockElementType.PARAGRAPH:
        return False
    para_token = element.tokens[0]
    if para_token.marker is None:
        return False
    para_tag = state.stylesheet.get_tag(para_token.marker)
    return para_tag.text_type != UsfmTextType.VERSE_TEXT and para_tag.text_type != UsfmTextType.NOT_SPECIFIED
