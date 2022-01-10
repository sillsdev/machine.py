from typing import Iterable

from ..detokenizer import Detokenizer


class SentencePieceDetokenizer(Detokenizer[str, str]):
    def detokenize(self, tokens: Iterable[str]) -> str:
        return "".join(tokens).replace("▁", " ").lstrip()
