from typing import Iterable

from ..scripture.verse_ref import Versification
from .dictionary_text_corpus import DictionaryTextCorpus
from .scripture_text import ScriptureText


class ScriptureTextCorpus(DictionaryTextCorpus):
    def __init__(self, versification: Versification, texts: Iterable[ScriptureText] = []) -> None:
        super().__init__(texts)
        self._versification = versification

    @property
    def versification(self) -> Versification:
        return self._versification
