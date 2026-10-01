from typing import Dict, List, Optional, Sequence, Tuple

import regex as re

from ..scripture.verse_ref import VerseRef, Versification
from .scripture_ref_usfm_parser_handler_base import ScriptureRefUsfmParserHandlerBase
from .usfm_parser_state import UsfmParserState
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer

_TRAILING_PARAGRAPH_MARKER_PATTERNS = re.compile(r"^(?:mte?\d*|ms\d*|sd?\d*|mr|sr|sp|d|r)$")


def _change_versification(verse_ref: VerseRef, versification: Versification) -> VerseRef:
    new_verse_ref = verse_ref.copy()
    new_verse_ref.change_versification(versification)
    return new_verse_ref


def _new_nb_token() -> UsfmToken:
    return UsfmToken(UsfmTokenType.PARAGRAPH, "nb", "", "", "")


class ConvertUsfmVersificationHandler(ScriptureRefUsfmParserHandlerBase):
    def __init__(self, target_versification: Versification) -> None:
        super().__init__()
        self._tokens: List[UsfmToken] = []
        self._trailing_verse_tokens: List[Tuple[int, UsfmToken]] = []
        self._prev_verse_ref = VerseRef()
        self._verse_boundary = 0
        self._target_versification = target_versification
        self._insert_chapter_index = -1
        self._skip = False

    @property
    def tokens(self) -> Sequence[UsfmToken]:
        return self._tokens

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        super().chapter(state, number, marker, alt_number, pub_number)
        self._process_tokens(state)
        vr = state.verse_ref.copy()
        # The versification of verse 0 cannot properly be changed
        vr.verse = "1"
        if not self._prev_verse_ref.is_default and (
            _change_versification(vr, self._target_versification).book != self._prev_verse_ref.book
            or vr.chapter_num == -1
        ):
            self._skip = True
        self._insert_chapter_index = len(self._tokens)

    def verse(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        super().verse(state, number, marker, alt_number, pub_number)

        verse_ref = state.verse_ref

        self._process_tokens(state)

        verse_refs = [_change_versification(vr, self._target_versification) for vr in state.verse_ref.all_verses()]

        if (
            self._prev_verse_ref.is_default
            or (
                verse_refs[0].book_num == self._prev_verse_ref.book_num
                and verse_refs[0].chapter_num != self._prev_verse_ref.chapter_num
            )
        ) and verse_refs[0].chapter_num != -1:
            new_chapter_token = UsfmToken(UsfmTokenType.CHAPTER, "c", "", "", verse_refs[0].chapter)

            if self._insert_chapter_index == -1:
                chapter_index = len(self._tokens)
                self._tokens.append(new_chapter_token)
                trailing_at_chapter = [t for i, t in self._trailing_verse_tokens if i == chapter_index]
                if len(trailing_at_chapter) == 0:
                    # The chapter break falls mid-paragraph, so the paragraph continues across it.
                    self._tokens.append(_new_nb_token())
                else:
                    # The trailing markers follow the new chapter and break the paragraph. If they do not open a
                    # paragraph of their own, the verse still needs one.
                    last_paragraph = next(
                        (t for t in reversed(trailing_at_chapter) if t.type == UsfmTokenType.PARAGRAPH), None
                    )
                    if last_paragraph is None or _TRAILING_PARAGRAPH_MARKER_PATTERNS.match(last_paragraph.marker or ""):
                        self._trailing_verse_tokens.append((chapter_index, _new_nb_token()))
                    self._trailing_verse_tokens = [
                        (i + 1 if i == chapter_index else i, t) for i, t in self._trailing_verse_tokens
                    ]
            else:
                self._tokens.insert(self._insert_chapter_index, new_chapter_token)
                self._trailing_verse_tokens = [
                    (i + 1 if i >= self._insert_chapter_index else i, t) for i, t in self._trailing_verse_tokens
                ]

        added_verse_text = False
        duplicate_verse = False

        start: Optional[str] = None
        for vr in verse_refs:
            if (not self._prev_verse_ref.is_default and vr.book != self._prev_verse_ref.book) or vr.chapter_num == -1:
                continue
            if start is not None:
                end = "-" + self._prev_verse_ref.verse if start != self._prev_verse_ref.verse else ""
                if self._prev_verse_ref.book_num == vr.book_num and self._prev_verse_ref.chapter_num != vr.chapter_num:
                    self._add_trailing_tokens()
                    if not duplicate_verse:
                        self._tokens.append(UsfmToken(UsfmTokenType.VERSE, "v", "", "", start + end))
                    added_verse_text = self._add_next_text_token(state, added_verse_text)
                    self._tokens.append(UsfmToken(UsfmTokenType.CHAPTER, "c", "", "", vr.chapter))
                    self._tokens.append(_new_nb_token())
                    start = vr.verse
                    duplicate_verse = False
                    self._prev_verse_ref = vr
                elif self._prev_verse_ref.verse_num + 1 != vr.verse_num:
                    self._add_trailing_tokens()
                    if not duplicate_verse:
                        self._tokens.append(UsfmToken(UsfmTokenType.VERSE, "v", "", "", start + end))
                    added_verse_text = self._add_next_text_token(state, added_verse_text)
                    start = vr.verse
                    duplicate_verse = False
                    self._prev_verse_ref = vr
                else:
                    # The duplicated verse was already written, so the range starts after it.
                    if duplicate_verse:
                        start = vr.verse
                        duplicate_verse = False
                    self._prev_verse_ref = vr
            else:
                start = vr.verse
                duplicate_verse = vr == self._prev_verse_ref
                self._prev_verse_ref = vr
            verse_ref = vr

        if start is not None:
            self._add_trailing_tokens()
            end = "-" + self._prev_verse_ref.verse if start != self._prev_verse_ref.verse else ""
            if not duplicate_verse:
                self._tokens.append(UsfmToken(UsfmTokenType.VERSE, "v", "", "", start + end))
            self._skip = False
            self._insert_chapter_index = -1
            self._prev_verse_ref = verse_ref
        else:
            self._skip = True
            # Markers that introduce a dropped verse would otherwise be flushed at the next kept verse.
            self._trailing_verse_tokens.clear()

    def end_usfm(self, state: UsfmParserState) -> None:
        super().end_usfm(state)
        self._process_tokens(state)
        token = state.token
        if (
            not self._skip
            and token is not None
            and not (token.type == UsfmTokenType.CHAPTER or token.type == UsfmTokenType.VERSE)
        ):
            self._tokens.append(token)

    def get_usfm(self, stylesheet: UsfmStylesheet) -> str:
        tokenizer = UsfmTokenizer(stylesheet)
        return tokenizer.detokenize(self._tokens)

    def _add_next_text_token(self, state: UsfmParserState, added_verse_text: bool) -> bool:
        if not added_verse_text and state.index + 1 < len(state.tokens):
            next_token = state.tokens[state.index + 1]
            if next_token.type == UsfmTokenType.TEXT:
                self._tokens.append(next_token)
                self._verse_boundary += 1
                return True
        return added_verse_text

    def _process_tokens(self, state: UsfmParserState) -> None:
        offset = 0
        in_preserved_paragraph = False
        while self._verse_boundary + offset < state.index:
            token = state.tokens[self._verse_boundary + offset]
            if _is_preserved_trailing_paragraph_marker(state.tokens, self._verse_boundary + offset):
                in_preserved_paragraph = True
            elif in_preserved_paragraph:
                in_preserved_paragraph = token.type != UsfmTokenType.PARAGRAPH
            else:
                in_preserved_paragraph = False

            if in_preserved_paragraph:
                self._trailing_verse_tokens.append((len(self._tokens), token))
            elif not self._skip:
                self._tokens.append(token)

            offset += 1
        self._verse_boundary = state.index + 1

    def _add_trailing_tokens(self) -> None:
        grouped: Dict[int, List[UsfmToken]] = {}
        for index, token in self._trailing_verse_tokens:
            grouped.setdefault(index, []).append(token)
        for index in sorted(grouped, reverse=True):
            self._tokens[index:index] = grouped[index]
        self._trailing_verse_tokens.clear()


def _is_preserved_trailing_paragraph_marker(tokens: Sequence[UsfmToken], index: int) -> bool:
    token = tokens[index]
    next_token = tokens[index + 1] if index + 1 < len(tokens) else None
    return (
        token.type == UsfmTokenType.PARAGRAPH
        and next_token is not None
        and (next_token.type == UsfmTokenType.VERSE or _is_preserved_trailing_paragraph_marker(tokens, index + 1))
    ) or (
        token.type == UsfmTokenType.PARAGRAPH
        and token.marker is not None
        and _TRAILING_PARAGRAPH_MARKER_PATTERNS.match(token.marker) is not None
    )
