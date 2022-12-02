from typing import Optional, Sequence, Tuple, Union

import regex as re

from ..scripture.canon import book_id_to_number
from ..scripture.verse_ref import Versification, VersificationType
from ..utils.typeshed import StrPath
from .usfm_parser_handler import UsfmParserHandler
from .usfm_parser_state import UsfmElementType, UsfmParserElement, UsfmParserState
from .usfm_stylesheet import UsfmStylesheet, is_cell_range
from .usfm_tag import UsfmTextType
from .usfm_token import UsfmToken, UsfmTokenType
from .usfm_tokenizer import UsfmTokenizer

_OPT_BREAK_SPLITTER = re.compile(r"(//)")


def parse_usfm(
    usfm: str,
    handler: UsfmParserHandler,
    stylesheet: Union[StrPath, UsfmStylesheet] = "usfm.sty",
    versification: Optional[Versification] = None,
    preserve_whitespace: bool = False,
) -> None:
    parser = UsfmParser(usfm, handler, stylesheet, versification, preserve_whitespace)
    parser.process_tokens()


class UsfmParser:
    def __init__(
        self,
        usfm: Union[str, Sequence[UsfmToken]],
        handler: Optional[UsfmParserHandler] = None,
        stylesheet: Union[StrPath, UsfmStylesheet] = "usfm.sty",
        versification: Optional[Versification] = None,
        tokens_preserve_whitespace: bool = False,
    ) -> None:
        if isinstance(stylesheet, UsfmStylesheet):
            self.stylesheet = stylesheet
        else:
            self.stylesheet = UsfmStylesheet(stylesheet)
        if isinstance(usfm, str):
            tokenizer = UsfmTokenizer(self.stylesheet)
            tokens = tokenizer.tokenize(usfm, preserve_whitespace=tokens_preserve_whitespace)
        else:
            tokens = usfm
        if versification is None:
            versification = Versification.get_builtin(VersificationType.ENGLISH)
        self.state = UsfmParserState(self.stylesheet, versification, tokens)
        self.handler = handler
        self.tokens_preserve_whitespace = tokens_preserve_whitespace
        self._skip = 0

    def process_tokens(self) -> None:
        while self.process_token():
            pass

    def process_token(self) -> bool:
        # If past end
        if self.state.index >= len(self.state.tokens) - 1:
            if self.handler is not None:
                self.handler.end_usfm(self.state)
            return False
        elif self.state.index < 0:
            if self.handler is not None:
                self.handler.start_usfm(self.state)

        # Move to next token
        self.state.index += 1

        # Update verse offset with previous token (since verse offset is from start of current token)
        if self.state.prev_token is not None:
            self.state.verse_offset += self.state.prev_token.get_length(add_spaces=not self.tokens_preserve_whitespace)

        # Skip over tokens that are to be skipped, ensuring that special_token state is True.
        if self._skip > 0:
            self._skip -= 1
            self.state.special_token = True
            return True

        # Reset special token and figure status
        self.state.special_token = False

        token = self.state.token

        assert token is not None
        # Switch unknown types to either character or paragraph
        token_type = token.type
        if token_type == UsfmTokenType.UNKNOWN:
            token_type = self._determine_unknown_token_type()

        if self.handler is not None and token.marker is not None and len(token.marker) > 0:
            self.handler.got_marker(self.state, token.marker)

        if token_type in {UsfmTokenType.BOOK, UsfmTokenType.CHAPTER}:
            self._close_all()
        elif token_type == UsfmTokenType.PARAGRAPH:
            if token.marker == "tr":
                # Handle special case of table rows
                while (
                    len(self.state.stack) > 0
                    and self.state.peek().type != UsfmElementType.TABLE
                    and self.state.peek().type != UsfmElementType.SIDEBAR
                ):
                    self._close_element()
            elif token.marker == "esb":
                # Handle special case of sidebars
                self._close_all()
            else:
                # Close all but sidebar
                while len(self.state.stack) > 0 and self.state.peek().type != UsfmElementType.SIDEBAR:
                    self._close_element()
        elif token_type == UsfmTokenType.CHARACTER:
            if self._is_cell(token):
                # Handle special case of table cell
                # Close until row
                while self.state.peek().type != UsfmElementType.ROW:
                    self._close_element()
            elif self._is_ref(token):
                # Handle refs
                # Refs don't close anything
                pass
            elif token.marker is not None and not token.marker.startswith("+"):
                # If non-nested character style, close all character styles
                self._close_char_styles()
        elif token_type == UsfmTokenType.VERSE:
            para_tag = self.state.para_tag
            if para_tag is not None and para_tag.text_type != UsfmTextType.VERSE_TEXT and para_tag.text_type != 0:
                self._close_all()
            else:
                self._close_note()
        elif token_type == UsfmTokenType.NOTE:
            self._close_note()
        elif token_type == UsfmTokenType.END:
            # If end marker for an active note
            if any(
                e.type == UsfmElementType.NOTE and e.marker is not None and e.marker + "*" == token.marker
                for e in self.state.stack
            ):
                self._close_note(closed=True)
            else:
                # If end marker for a character style on stack, close it
                # If no matching end marker, close all character styles on top of stack
                unmatched = True
                while len(self.state.stack) > 0:
                    elem = self.state.peek()
                    if elem.type != UsfmElementType.CHAR:
                        break

                    # Determine if a + prefix is needed to close it (was nested char style)
                    plus_prefix = len(self.state.stack) > 1 and self.state.stack[-2].type == UsfmElementType.CHAR

                    assert elem.marker is not None
                    if ("+" if plus_prefix else "") + elem.marker + "*" == token.marker:
                        self._close_element(closed=True)
                        unmatched = False
                        break
                    else:
                        self._close_element()

                # Unmatched end marker
                if unmatched and self.handler is not None:
                    assert token.marker is not None
                    self.handler.unmatched(self.state, token.marker)

        # Handle tokens
        if token_type == UsfmTokenType.BOOK:
            assert token.marker is not None
            self.state.push(UsfmParserElement(UsfmElementType.BOOK, token.marker))

            # Code is always upper case
            assert token.data is not None
            code = token.data.upper()

            # Update verse ref. Leave book alone if not empty to prevent parsing errors on books with bad id lines.
            verse_ref = self.state.verse_ref
            if verse_ref.book == "" and book_id_to_number(code) != 0:
                verse_ref.book = code
            verse_ref.chapter_num = 1
            verse_ref.verse_num = 0
            self.state.verse_offset = 0

            # Book start.
            if self.handler is not None:
                self.handler.start_book(self.state, token.marker, code)
        elif token_type == UsfmTokenType.CHAPTER:
            assert token.marker is not None
            # Get alternate chapter number
            alt_chapter: Optional[str] = None
            if self.state.index < len(self.state.tokens) - 3:
                alt_chapter_token = self.state.tokens[self.state.index + 1]
                alt_chapter_num_token = self.state.tokens[self.state.index + 2]
                alt_chapter_end_token = self.state.tokens[self.state.index + 3]
                if (
                    alt_chapter_token.marker == "ca"
                    and alt_chapter_num_token.text is not None
                    and alt_chapter_end_token.marker == "ca*"
                ):
                    alt_chapter = alt_chapter_num_token.text.strip()
                    self._skip += 3

                    # Skip blank space after if present
                    if self.state.index + self._skip < len(self.state.tokens) - 1:
                        blank_token = self.state.tokens[self.state.index + self._skip + 1]
                        if blank_token.text is not None and len(blank_token.text.strip()) == 0:
                            self._skip += 1

            # Get publishable chapter number
            pub_chapter: Optional[str] = None
            if self.state.index + self._skip < len(self.state.tokens) - 2:
                pub_chapter_token = self.state.tokens[self.state.index + self._skip + 1]
                pub_chapter_num_token = self.state.tokens[self.state.index + self._skip + 2]
                if pub_chapter_token.marker == "cp" and pub_chapter_num_token.text is not None:
                    pub_chapter = pub_chapter_num_token.text.strip()
                    self._skip += 2

            assert token.data is not None
            verse_ref = self.state.verse_ref
            verse_ref.chapter = token.data
            verse_ref.verse_num = 0
            # Verse offset is not zeroed for chapter 1, as it is part of intro
            if verse_ref.chapter_num != 1:
                self.state.verse_offset = 0

            if self.handler is not None:
                self.handler.chapter(self.state, token.data, token.marker, alt_chapter, pub_chapter)
        elif token_type == UsfmTokenType.VERSE:
            assert token.marker is not None
            # Get alternate verse number
            alt_verse: Optional[str] = None
            if self.state.index < len(self.state.tokens) - 3:
                alt_verse_token = self.state.tokens[self.state.index + 1]
                alt_verse_num_token = self.state.tokens[self.state.index + 2]
                alt_verse_end_token = self.state.tokens[self.state.index + 3]
                if (
                    alt_verse_token.marker == "va"
                    and alt_verse_num_token.text is not None
                    and alt_verse_end_token.marker == "va*"
                ):
                    alt_verse = alt_verse_num_token.text.strip()
                    self._skip += 3

            # Get publishable verse number
            pub_verse: Optional[str] = None
            if self.state.index + self._skip < len(self.state.tokens) - 3:
                pub_verse_token = self.state.tokens[self.state.index + self._skip + 1]
                pub_verse_num_token = self.state.tokens[self.state.index + self._skip + 2]
                pub_verse_end_token = self.state.tokens[self.state.index + self._skip + 3]
                if (
                    pub_verse_token.marker == "vp"
                    and pub_verse_num_token.text is not None
                    and pub_verse_end_token.marker == "vp*"
                ):
                    pub_chapter = pub_verse_num_token.text.strip()
                    self._skip += 3

            assert token.data is not None
            verse_ref = self.state.verse_ref
            verse_ref.verse = token.data
            self.state.verse_offset = 0

            if self.handler is not None:
                self.handler.verse(self.state, token.data, token.marker, alt_verse, pub_verse)
        elif token_type == UsfmTokenType.PARAGRAPH:
            assert token.marker is not None
            if token.marker == "tr":
                # Handle special case of table rows
                # Start table if not open
                if all(e.type != UsfmElementType.TABLE for e in self.state.stack):
                    self.state.push(UsfmParserElement(UsfmElementType.TABLE, None))
                    if self.handler is not None:
                        self.handler.start_table(self.state)

                self.state.push(UsfmParserElement(UsfmElementType.ROW, token.marker))

                # Row start
                if self.handler is not None:
                    self.handler.start_row(self.state, token.marker)
            elif token.marker == "esb":
                # Handle special case of sidebars
                self.state.push(UsfmParserElement(UsfmElementType.SIDEBAR, token.marker))

                # Look for category
                category: Optional[str] = None
                if self.state.index < len(self.state.tokens) - 3:
                    category_token = self.state.tokens[self.state.index + 1]
                    category_value_token = self.state.tokens[self.state.index + 2]
                    category_end_token = self.state.tokens[self.state.index + 3]
                    if (
                        category_token.marker == "esbc"
                        and category_value_token.text is not None
                        and category_end_token.marker == "esbc*"
                    ):
                        category = category_value_token.text.strip()
                        self._skip += 3

                if self.handler is not None:
                    self.handler.start_sidebar(self.state, token.marker, category)
            elif token.marker == "esbe":
                # Close sidebar if in sidebar
                if any(e.type == UsfmElementType.SIDEBAR for e in self.state.stack):
                    while len(self.state.stack) > 0:
                        self._close_element(self.state.peek().type == UsfmElementType.SIDEBAR)
                elif self.handler is not None:
                    self.handler.unmatched(self.state, token.marker)
            else:
                self.state.push(UsfmParserElement(UsfmElementType.PARA, token.marker))

                # Paragraph start
                if self.handler is not None:
                    self.handler.start_para(
                        self.state, token.marker, token.type == UsfmTokenType.UNKNOWN, token.attributes
                    )
        elif token_type == UsfmTokenType.CHARACTER:
            assert token.marker is not None
            if self._is_cell(token):
                # Handle special case of table cells (treated as special character style)
                align = "start"
                if len(token.marker) > 2 and token.marker[2] == "c":
                    align = "center"
                elif len(token.marker) > 2 and token.marker[2] == "r":
                    align = "end"

                _, base_marker, col_span = is_cell_range(token.marker)
                self.state.push(UsfmParserElement(UsfmElementType.CELL, base_marker))

                if self.handler is not None:
                    self.handler.start_cell(self.state, base_marker, align, col_span)
            elif self._is_ref(token):
                # xrefs are special tokens (they do not stand alone)
                self.state.special_token = True

                display, target = self._parse_display_and_target()

                self._skip += 2

                if self.handler is not None:
                    self.handler.ref(self.state, token.marker, display, target)
            else:
                invalid_marker = False
                if token.marker.startswith("+"):
                    # Only strip + if properly nested
                    char_tag = self.state.char_tag
                    actual_marker = token.marker.lstrip("+") if char_tag is not None else token.marker
                    invalid_marker = char_tag is None
                else:
                    actual_marker = token.marker

                self.state.push(UsfmParserElement(UsfmElementType.CHAR, actual_marker, token.attributes))
                if self.handler is not None:
                    self.handler.start_char(
                        self.state,
                        actual_marker,
                        token.type == UsfmTokenType.UNKNOWN or invalid_marker,
                        token.attributes,
                    )
        elif token_type == UsfmTokenType.NOTE:
            assert token.marker is not None
            assert token.data is not None
            # Look for category
            category: Optional[str] = None
            if self.state.index < len(self.state.tokens) - 3:
                category_token = self.state.tokens[self.state.index + 1]
                category_value_token = self.state.tokens[self.state.index + 2]
                category_end_token = self.state.tokens[self.state.index + 3]
                if (
                    category_token.marker == "cat"
                    and category_value_token.text is not None
                    and category_end_token.marker == "cat*"
                ):
                    category = category_value_token.text.strip()
                    self._skip += 3

            self.state.push(UsfmParserElement(UsfmElementType.NOTE, token.marker))

            if self.handler is not None:
                self.handler.start_note(self.state, token.marker, token.data, category)
        elif token_type == UsfmTokenType.TEXT:
            text = token.text
            assert text is not None
            if (
                (
                    self.state.index == len(self.state.tokens) - 1
                    or self.state.tokens[self.state.index + 1].type
                    in {UsfmTokenType.PARAGRAPH, UsfmTokenType.BOOK, UsfmTokenType.CHAPTER}
                )
                and len(text) > 0
                and text[-1] == " "
            ):
                text = text[:-1]

            if self.handler is not None:
                # Replace ~ with nbsp
                text = text.replace("~", "\u00A0")

                # Replace // with <optbreak/>
                for part in _OPT_BREAK_SPLITTER.split(text):
                    if part == "//":
                        self.handler.opt_break(self.state)
                    else:
                        self.handler.text(self.state, part)
        elif token_type in {UsfmTokenType.MILESTONE, UsfmTokenType.MILESTONE_END}:
            assert token.marker is not None
            # currently, parse state doesn't need to be update, so just inform the handler about the milestone.
            if self.handler is not None:
                self.handler.milestone(
                    self.state, token.marker, token.type == UsfmTokenType.MILESTONE, token.attributes
                )
        return True

    def _parse_display_and_target(self) -> Tuple[str, str]:
        next_token = self.state.tokens[self.state.index + 1]
        assert next_token.text is not None
        index = next_token.text.find("|")
        display = next_token.text[:index]
        target = next_token.text[index + 1 :]
        return display, target

    def _close_all(self) -> None:
        while len(self.state.stack) > 0:
            self._close_element()

    def _is_study_bible_item_closed(self, start_marker: str, ending_marker: str) -> bool:
        for i in range(self.state.index + 1, len(self.state.tokens)):
            token = self.state.tokens[i]
            if token.marker == ending_marker:
                return True

            if token.marker == start_marker or token.type in {UsfmTokenType.BOOK, UsfmTokenType.CHAPTER}:
                return False
        return False

    def _determine_unknown_token_type(self) -> UsfmTokenType:
        if any(e.type == UsfmElementType.NOTE for e in self.state.stack):
            return UsfmTokenType.CHARACTER
        return UsfmTokenType.PARAGRAPH

    def _close_note(self, closed: bool = False) -> None:
        if any(elem.type == UsfmElementType.NOTE for elem in self.state.stack):
            elem: Optional[UsfmParserElement] = None
            while len(self.state.stack) > 0 and (elem is None or elem.type != UsfmElementType.NOTE):
                elem = self.state.peek()
                self._close_element(closed and elem.type == UsfmElementType.NOTE)

    def _close_char_styles(self) -> None:
        while len(self.state.stack) > 0 and self.state.peek().type == UsfmElementType.CHAR:
            self._close_element()

    def _close_element(self, closed: bool = False) -> None:
        element = self.state.pop()
        if self.handler is not None:
            if element.type == UsfmElementType.BOOK:
                assert element.marker is not None
                self.handler.end_book(self.state, element.marker)
            elif element.type == UsfmElementType.PARA:
                assert element.marker is not None
                self.handler.end_para(self.state, element.marker)
            elif element.type == UsfmElementType.CHAR:
                assert element.marker is not None
                self.handler.end_char(self.state, element.marker, element.attributes, closed)
            elif element.type == UsfmElementType.NOTE:
                assert element.marker is not None
                self.handler.end_note(self.state, element.marker, closed)
            elif element.type == UsfmElementType.TABLE:
                self.handler.end_table(self.state)
            elif element.type == UsfmElementType.ROW:
                assert element.marker is not None
                self.handler.end_row(self.state, element.marker)
            elif element.type == UsfmElementType.CELL:
                assert element.marker is not None
                self.handler.end_cell(self.state, element.marker)
            elif element.type == UsfmElementType.SIDEBAR:
                assert element.marker is not None
                self.handler.end_sidebar(self.state, element.marker, closed)

    def _is_cell(self, token: UsfmToken) -> bool:
        return (
            token.type == UsfmTokenType.CHARACTER
            and token.marker is not None
            and (token.marker.startswith("th") or token.marker.startswith("tc"))
            and any(elem.type == UsfmElementType.ROW for elem in self.state.stack)
        )

    def _is_ref(self, token: UsfmToken) -> bool:
        if token.marker != "ref":
            return False

        if self.state.index >= len(self.state.tokens) - 2:
            return False

        attr_token = self.state.tokens[self.state.index + 1]
        if attr_token.text is None or "|" not in attr_token.text:
            return False

        end_token = self.state.tokens[self.state.index + 2]
        return end_token.type == UsfmTokenType.END and end_token.marker == token.end_marker
