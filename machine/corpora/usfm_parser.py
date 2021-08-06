from typing import List, Sequence, Tuple

from .usfm_marker import UsfmStyleType, UsfmTextProperties
from .usfm_stylesheet import UsfmStylesheet
from .usfm_token import UsfmToken, UsfmTokenType


class UsfmParser:
    def __init__(self, stylesheet: UsfmStylesheet) -> None:
        self._stylesheet = stylesheet

    def parse(self, usfm: str, preserve_whitespace: bool = False) -> Sequence[UsfmToken]:
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

                tokens.append(UsfmToken(UsfmTokenType.TEXT, None, text))

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

            marker_str = usfm[marker_start:index].rstrip()

            # Multiple whitespace after non-end marker is ok
            if not marker_str.endswith("*") and not preserve_whitespace:
                while index < len(usfm) and _is_nonsemantic_whitespace(usfm[index]):
                    index += 1

            # Lookup marker
            marker = self._stylesheet.get_marker(marker_str.lstrip("+"))

            # If starts with a plus and is not a character style, it is an unknown marker
            if marker_str.startswith("+") and marker.style_type != UsfmStyleType.CHARACTER:
                marker = self._stylesheet.get_marker(marker_str)

            if marker.style_type == UsfmStyleType.CHARACTER:
                if (marker.text_properties & UsfmTextProperties.VERSE) == UsfmTextProperties.VERSE:
                    index, text = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.VERSE, marker, text))
                else:
                    tokens.append(UsfmToken(UsfmTokenType.CHARACTER, marker, None))
            elif marker.style_type == UsfmStyleType.PARAGRAPH:
                # Handle chapter special case
                if (marker.text_properties & UsfmTextProperties.CHAPTER) == UsfmTextProperties.CHAPTER:
                    index, text = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.CHAPTER, marker, text))
                elif (marker.text_properties & UsfmTextProperties.BOOK) == UsfmTextProperties.BOOK:
                    index, text = _get_next_word(usfm, index, preserve_whitespace)
                    tokens.append(UsfmToken(UsfmTokenType.BOOK, marker, text))
                else:
                    tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker, None))
            elif marker.style_type == UsfmStyleType.NOTE:
                index, text = _get_next_word(usfm, index, preserve_whitespace)
                tokens.append(UsfmToken(UsfmTokenType.NOTE, marker, text))
            elif marker.style_type == UsfmStyleType.END:
                tokens.append(UsfmToken(UsfmTokenType.END, marker, None))
            elif marker.style_type == UsfmStyleType.UNKNOWN:
                # End tokens are always end tokens, even if unknown
                if marker_str.endswith("*"):
                    tokens.append(UsfmToken(UsfmTokenType.END, marker, None))
                # Handle special case of esb and esbe which might not be in basic stylesheet but are always sidebars
                # and so should be tokenized as paragraphs
                elif marker_str == "esb" or marker_str == "esbe":
                    tokens.append(UsfmToken(UsfmTokenType.PARAGRAPH, marker, None))
                else:
                    # Create unknown token with a corresponding end note
                    tokens.append(UsfmToken(UsfmTokenType.UNKNOWN, marker, None))

        # Forces a space to be present in tokenization if immediately before a token requiring a preceeding CR/LF. This
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
                            tokens[i - 1] = UsfmToken(UsfmTokenType.TEXT, None, prev_token.text + " ")
                    elif prev_token.type == UsfmTokenType.END:
                        # Insert space token after * of end marker
                        tokens.insert(i, UsfmToken(UsfmTokenType.TEXT, None, " "))
                        i += 1

        return tokens


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
