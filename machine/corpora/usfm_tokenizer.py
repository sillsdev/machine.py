from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import regex as re

from ..utils.typeshed import StrPath
from .usfm_stylesheet import UsfmStylesheet
from .usfm_tag import UsfmStyleType, UsfmTextProperties
from .usfm_token import UsfmToken, UsfmTokenType

_RTL_VERSE_REGEX = re.compile(r"[\u200E\u200F]*(\d+\w?)[\u200E\u200F]*([\p{P}\p{S}])[\u200E\u200F]*(?=\d)")


class RtlReferenceOrder(Enum):
    NOT_SET = auto()
    BOOK_CHAPTER_VERSE = auto()
    BOOK_VERSE_CHAPTER = auto()


class UsfmTokenizer:
    def __init__(
        self,
        stylesheet: Union[StrPath, UsfmStylesheet] = "usfm.sty",
        rtl_reference_order: RtlReferenceOrder = RtlReferenceOrder.NOT_SET,
    ) -> None:
        if isinstance(stylesheet, UsfmStylesheet):
            self.stylesheet = stylesheet
        else:
            self.stylesheet = UsfmStylesheet(stylesheet)
        self.rtl_reference_order = rtl_reference_order

    def tokenize(self, usfm: str, preserve_whitespace: bool = False) -> Sequence[UsfmToken]:
        tokens: List[UsfmToken] = []

        index = 0
        while index < len(usfm):
            next_marker_index = usfm.find("\\", index + 1) if index < len(usfm) - 1 else -1
            if next_marker_index == -1:
                next_marker_index = len(usfm)

            # If text, create text token until end or next \
            ch = usfm[index]
            if ch != "\\":
                text = usfm[index:next_marker_index]
                if not preserve_whitespace:
                    text = _regularize_spaces(text)

                attribute_token, text = self._handle_attributes(
                    usfm, preserve_whitespace, tokens, next_marker_index, text
                )

                if len(text) > 0:
                    tokens.append(UsfmToken(UsfmTokenType.TEXT, None, text, None))

                if attribute_token is not None:
                    tokens.append(attribute_token)

                index = next_marker_index
                continue

            # Get marker (and move past whitespace or star ending)
            index += 1
            marker_start = index
            while index < len(usfm):
                ch = usfm[index]

                # Backslash starts a new marker
                if ch == "\\":
                    break

                # don't require a space before the | that starts attributes - mainly for milestones to allow
                # \qt-s|speaker\*
                if ch == "|":
                    break

                # End star is part of marker
                if ch == "*":
                    index += 1
                    break

                if _is_nonsemantic_whitespace(ch):
                    # Preserve whitespace if needed, otherwise skip
                    if not preserve_whitespace:
                        index += 1
                    break
                index += 1

            marker = usfm[marker_start:index].rstrip()
            # Milestone stop/end markers are ended with \*, so marker will just be * and can be skipped
            if marker == "*":
                # make sure that previous token was a milestone - have to skip space only tokens that may have been
                # added when preserve_whitespace is true.
                prev_token: Optional[UsfmToken] = None
                if len(tokens) > 0:
                    prev_token = next(
                        t
                        for t in reversed(tokens)
                        if t.type != UsfmTokenType.TEXT or (t.text is not None and t.text.strip() != "")
                    )
                if (
                    prev_token is not None
                    and prev_token.marker is not None
                    and prev_token.type in {UsfmTokenType.MILESTONE, UsfmTokenType.MILESTONE_END}
                ):
                    # if the last item is an empty text token, remove it so we don't get extra space.
                    if tokens[-1].type == UsfmTokenType.TEXT:
                        tokens.pop()
                    continue

            # Multiple whitespace after non-end marker is ok
            if not marker.endswith("*") and not preserve_whitespace:
                while index < len(usfm) and _is_nonsemantic_whitespace(usfm[index]):
                    index += 1

            # Lookup marker
            tag = self.stylesheet.get_tag(marker.lstrip("+"))

            # If starts with a plus and is not a character style or an end style, it is an unknown tag
            if marker.startswith("+") and tag.style_type not in {UsfmStyleType.CHARACTER, UsfmStyleType.END}:
                tag = self.stylesheet.get_tag(marker)

            end_marker = marker + "*" if tag.style_type != UsfmStyleType.MILESTONE else tag.end_marker

            if tag.style_type == UsfmStyleType.CHARACTER:
                if (tag.text_properties & UsfmTextProperties.VERSE) == UsfmTextProperties.VERSE:
                    index, data = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.VERSE, marker, None, None, data))
                else:
                    tokens.append(UsfmToken(UsfmTokenType.CHARACTER, marker, None, end_marker))
            elif tag.style_type == UsfmStyleType.PARAGRAPH:
                # Handle chapter special case
                if (tag.text_properties & UsfmTextProperties.CHAPTER) == UsfmTextProperties.CHAPTER:
                    index, data = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.CHAPTER, marker, None, None, data))
                elif (tag.text_properties & UsfmTextProperties.BOOK) == UsfmTextProperties.BOOK:
                    index, data = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.BOOK, marker, None, None, data))
                else:
                    tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker, None, end_marker))
            elif tag.style_type == UsfmStyleType.NOTE:
                index, data = _get_next_word(usfm, index, preserve_whitespace)
                tokens.append(UsfmToken(UsfmTokenType.NOTE, marker, None, end_marker, data))
            elif tag.style_type == UsfmStyleType.END:
                tokens.append(UsfmToken(UsfmTokenType.END, marker, None, None))
            elif tag.style_type == UsfmStyleType.UNKNOWN:
                # End tokens are always end tokens, even if unknown
                if marker.endswith("*"):
                    tokens.append(UsfmToken(UsfmTokenType.END, marker, None, None))
                # Handle special case of esb and esbe which might not be in basic stylesheet but are always sidebars
                # and so should be tokenized as paragraphs
                elif marker == "esb" or marker == "esbe":
                    tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker, None, end_marker))
                else:
                    # Create unknown token with a corresponding end note
                    tokens.append(UsfmToken(UsfmTokenType.UNKNOWN, marker, None, marker + "*"))
            elif tag.style_type in {UsfmStyleType.MILESTONE, UsfmStyleType.MILESTONE_END}:
                # if a milestone is not followed by a ending \* treat don't create a milestone token for the begining.
                # Instead create at text token for all the text up to the beginning of the next marker. This will make
                # typing of milestones easiest since the partially typed milestone more be reformatted to have a normal
                # ending even if it hasn't been typed yet.
                if not _milestone_ended(usfm, index):
                    end_of_text = usfm.find("\\", index) if index < len(usfm) - 1 else -1
                    if end_of_text == -1:
                        end_of_text = len(usfm)
                    milestone_text = usfm[index:end_of_text]
                    # add back space that was removed after marker
                    if len(milestone_text) > 0 and milestone_text[0] not in {" ", "|"}:
                        milestone_text = " " + milestone_text
                    tokens.append(UsfmToken(UsfmTokenType.TEXT, None, "\\" + marker + milestone_text, None))
                    index = end_of_text
                elif tag.style_type == UsfmStyleType.MILESTONE:
                    tokens.append(UsfmToken(UsfmTokenType.MILESTONE, marker, None, end_marker))
                else:
                    tokens.append(UsfmToken(UsfmTokenType.MILESTONE_END, marker, None, None))

        # Forces a space to be present in tokenization if immediately before a token requiring a preceding CR/LF. This
        # is to ensure that when written to disk and re-read, that tokenization will match. For example,
        # "\p test\p here" requires a space after "test". Also, "\p \em test\em*\p here" requires a space token inserted
        # after \em*
        if not preserve_whitespace:
            for i in range(1, len(tokens)):
                cur_token = tokens[i]
                prev_token = tokens[i - 1]
                # If requires newline (verses do, except when after '(' or '[')
                if (
                    cur_token.type == UsfmTokenType.BOOK
                    or cur_token.type == UsfmTokenType.CHAPTER
                    or cur_token.type == UsfmTokenType.PARAGRAPH
                    or (
                        cur_token.type == UsfmTokenType.VERSE
                        and not (
                            prev_token.type == UsfmTokenType.TEXT
                            and prev_token.text is not None
                            and (prev_token.text.endswith("(") or prev_token.text.endswith("["))
                        )
                    )
                ):
                    # Add space to text token
                    if prev_token.type == UsfmTokenType.TEXT:
                        assert prev_token.text is not None
                        if not prev_token.text.endswith(" "):
                            t = tokens[i - 1]
                            assert t.text is not None
                            t.text = t.text + " "
                    elif prev_token.type == UsfmTokenType.END:
                        # Insert space token after * of end marker
                        tokens.insert(i, UsfmToken(UsfmTokenType.TEXT, None, " ", None))
                        i += 1

        return tokens

    def detokenize(self, tokens: Iterable[UsfmToken], tokens_have_whitespace: bool = False) -> str:
        prev_token: Optional[UsfmToken] = None
        usfm = ""
        for token in tokens:
            token_usfm = ""
            if token.type in {UsfmTokenType.BOOK, UsfmTokenType.CHAPTER, UsfmTokenType.PARAGRAPH}:
                # Strip space from end of string before CR/LF
                if len(usfm) > 0:
                    if (
                        usfm[-1] == " "
                        and (prev_token is not None and prev_token.to_usfm().strip() != "")
                        or not tokens_have_whitespace
                    ):
                        usfm = usfm[:-1]
                    if not tokens_have_whitespace:
                        usfm += "\r\n"
                token_usfm = token.to_usfm()
            elif token.type is UsfmTokenType.VERSE:
                # Add newline if after anything other than [ or (
                if len(usfm) > 0 and usfm[-1] != "[" and usfm[-1] != "(":
                    if (
                        usfm[-1] == " "
                        and (prev_token is not None and prev_token.to_usfm().strip() != "")
                        or not tokens_have_whitespace
                    ):
                        usfm = usfm[:-1]
                    if not tokens_have_whitespace:
                        usfm += "\r\n"

                token_usfm = token.to_usfm().strip() if tokens_have_whitespace else token.to_usfm()

                # want RTL mark around all punctuation in verses
                if self.rtl_reference_order is not RtlReferenceOrder.NOT_SET:
                    direction_marker = (
                        "\u200e" if self.rtl_reference_order is RtlReferenceOrder.BOOK_VERSE_CHAPTER else "\u200f"
                    )
                    token_usfm = _RTL_VERSE_REGEX.sub(token_usfm, f"$1{direction_marker}$2")

            elif token.type is UsfmTokenType.TEXT:
                # Ensure spaces are preserved
                token_usfm = token.to_usfm()
                if tokens_have_whitespace and len(usfm) > 0 and usfm[-1] == " ":
                    if (
                        len(token_usfm) > 0
                        and token_usfm[0] == " "
                        and prev_token is not None
                        and prev_token.to_usfm().strip() != ""
                    ) or token_usfm.startswith("\r\n"):
                        usfm = usfm[:-1]
                    else:
                        token_usfm = token_usfm.lstrip(" ")
            else:
                token_usfm = token.to_usfm()

            usfm += token_usfm
            prev_token = token

        # Make sure begins without space or CR/LF
        if len(usfm) > 0 and usfm[0] == " ":
            usfm = usfm[1:]
        if len(usfm) > 0 and usfm[0] == "\r":
            usfm = usfm[2:]

        # Make sure ends without space and with a CR/LF
        if len(usfm) > 0 and usfm[-1] == " ":
            usfm = usfm[:-1]
        if len(usfm) > 0 and usfm[-1] != "\n":
            usfm += "\r\n"
        if len(usfm) > 3 and usfm[-3] == " " and usfm[-2] == "\r" and usfm[-1] == "\n":
            usfm = usfm[:-3] + usfm[-2:]
        return usfm

    def _handle_attributes(
        self, usfm: str, preserve_whitespace: bool, tokens: List[UsfmToken], next_marker_index: int, text: str
    ) -> Tuple[Optional[UsfmToken], str]:
        attribute_index = text.find("|")
        if attribute_index == -1:
            return None, text

        matching_token = _find_matching_start_marker(usfm, tokens, next_marker_index)
        if matching_token is None or matching_token.marker is None:
            return None, text

        assert matching_token.nestless_marker is not None
        matching_tag = self.stylesheet.get_tag(matching_token.nestless_marker)
        if matching_tag.style_type not in {
            UsfmStyleType.CHARACTER,
            UsfmStyleType.MILESTONE,
            UsfmStyleType.MILESTONE_END,
        }:
            return None, text  # leave attributes of other styles as regular text

        attribute_token: Optional[UsfmToken] = None
        attributes_value = text[attribute_index + 1 :]
        adjusted_text = matching_token.set_attributes(
            attributes_value,
            matching_tag.default_attribute_name,
            text[:attribute_index],
            preserve_whitespace,
        )
        if adjusted_text is not None:
            text = adjusted_text

            if matching_tag.style_type == UsfmStyleType.CHARACTER:  # Don't do this for milestones
                attribute_token = UsfmToken(UsfmTokenType.ATTRIBUTE, matching_tag.marker, None, None, attributes_value)
                attribute_token.copy_attributes(matching_token)
        return attribute_token, text


_ZERO_WIDTH_SPACE = "\u200B"


def _get_next_word(usfm: str, index: int, preserve_whitespace: bool) -> Tuple[int, str]:
    # Skip over leading spaces
    while index < len(usfm) and _is_nonsemantic_whitespace(usfm[index]):
        index += 1

    data_start = index
    while index < len(usfm) and not _is_nonsemantic_whitespace(usfm[index]) and usfm[index] != "\\":
        index += 1

    data = usfm[data_start:index]

    # Skip over trailing spaces
    if not preserve_whitespace:
        while index < len(usfm) and _is_nonsemantic_whitespace(usfm[index]):
            index += 1

    return index, data


def _is_nonsemantic_whitespace(c: str) -> bool:
    # Checks if is whitespace, but not U+3000 (IDEOGRAPHIC SPACE).
    return (c != "\u3000" and c.isspace()) or c == _ZERO_WIDTH_SPACE


def _regularize_spaces(text: str) -> str:
    was_space = False
    result = ""
    for i in range(len(text)):
        ch = text[i]
        # Control characters and CR/LF and TAB become spaces
        if ord(ch) < 32:
            if not was_space:
                result += " "
            was_space = True
        elif (
            not was_space and ch == _ZERO_WIDTH_SPACE and i + 1 < len(text) and _is_nonsemantic_whitespace(text[i + 1])
        ):
            # ZWSP is redundant if followed by a space
            pass
        elif _is_nonsemantic_whitespace(ch):
            # Keep other kinds of spaces
            if not was_space:
                result += ch
            was_space = True
        else:
            result += ch
            was_space = False
    return result


def _find_matching_start_marker(usfm: str, tokens: List[UsfmToken], next_marker_index: int) -> Optional[UsfmToken]:
    expected_start_marker = _before_end_marker(usfm, next_marker_index)
    if expected_start_marker is None:
        return None

    if expected_start_marker == "" and tokens[-1].type in {UsfmTokenType.MILESTONE, UsfmTokenType.MILESTONE_END}:
        return tokens[-1]

    nesting_level = 0
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        if token.type == UsfmTokenType.END:
            nesting_level += 1
        elif token.type not in {UsfmTokenType.TEXT, UsfmTokenType.ATTRIBUTE}:
            if nesting_level > 0:
                nesting_level -= 1
            elif nesting_level == 0:
                return token
    return None


def _before_end_marker(usfm: str, next_marker_index: int) -> Optional[str]:
    index = next_marker_index + 1
    while index < len(usfm) and usfm[index] != "*" and not usfm[index].isspace():
        index += 1

    if index >= len(usfm) or usfm[index] != "*":
        return None
    start_marker = usfm[next_marker_index + 1 : index - next_marker_index - 1]
    return start_marker


def _milestone_ended(usfm: str, index: int) -> bool:
    next_marker_index = usfm.find("\\", index) if index < len(usfm) else -1
    if next_marker_index == -1 or next_marker_index > len(usfm) - 2:
        return False

    return usfm[next_marker_index : next_marker_index + 2] == "\\*"
