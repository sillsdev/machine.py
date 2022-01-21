from typing import Iterable, Optional, Sequence

from .usx_token import UsxToken


class UsxVerse:
    def __init__(self, chapter: str, verse: str, is_sentence_start: bool, tokens: Iterable[UsxToken]) -> None:
        self._chapter = chapter
        self._verse = verse
        self._is_sentence_start = is_sentence_start
        self._tokens = list(tokens)

        prev_token: Optional[UsxToken] = None
        text = ""
        ends_with_space = False
        for token in self._tokens:
            if token.element is not None:
                if token.element.tag == "figure" and not ends_with_space:
                    text += " "
                if token.element.get("style") == "rq":
                    text = text.rstrip()

            if len(token.text) == 0 or token.text.startswith("\n"):
                continue

            if (
                prev_token is not None
                and token.para_element != prev_token.para_element
                and len(text) > 0
                and not ends_with_space
            ):
                text += " "

            text += str(token)
            ends_with_space = str(token).endswith(" ")
            prev_token = token
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
