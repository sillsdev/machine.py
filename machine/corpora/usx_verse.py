import xml.etree.ElementTree as etree
from typing import Iterable, Optional, Sequence

from .usx_token import UsxToken


class UsxVerse:
    def __init__(self, chapter: str, verse: str, is_sentence_start: bool, tokens: Iterable[UsxToken]) -> None:
        self._chapter = chapter
        self._verse = verse
        self._is_sentence_start = is_sentence_start
        self._tokens = list(tokens)

        prev_para_elem: Optional[etree.Element] = None
        text = ""
        for token in self._tokens:
            if len(token.text) == 0 or token.text.isspace():
                continue

            if token.para_element != prev_para_elem and len(text) > 0 and not text.endswith(" "):
                text += " "

            text += str(token)
            prev_para_elem = token.para_element
        self._text = text.strip()

    @property
    def chapter(self) -> str:
        return self._chapter

    @property
    def verse(self) -> str:
        return self._verse

    @property
    def is_sentence_start(self) -> bool:
        return self._is_sentence_start

    @property
    def tokens(self) -> Sequence[UsxToken]:
        return self._tokens

    @property
    def text(self) -> str:
        return self._text
