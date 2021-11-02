from typing import Iterable

from ..scripture.verse_ref import Versification
from ..tokenization.tokenizer import Tokenizer
from .dictionary_text_corpus import DictionaryTextCorpus
from .null_scripture_text import NullScriptureText
from .scripture_text import ScriptureText
from .text import Text


class ScriptureTextCorpus(DictionaryTextCorpus):
    def __init__(
        self, word_tokenizer: Tokenizer[str, int, str], versification: Versification, texts: Iterable[ScriptureText]
    ) -> None:
        super().__init__(texts)
        self._word_tokenizer = word_tokenizer
        self._versification = versification

    @property
    def versification(self) -> Versification:
        return self._versification

    @property
    def word_tokenizer(self) -> Tokenizer[str, int, str]:
        return self._word_tokenizer

    def create_null_text(self, id: str) -> Text:
        return NullScriptureText(self.word_tokenizer, id, self.versification)
