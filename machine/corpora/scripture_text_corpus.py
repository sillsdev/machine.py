from abc import abstractmethod

from ..scripture.verse_ref import Versification
from .corpora_helpers import get_scripture_text_sort_key
from .dictionary_text_corpus import DictionaryTextCorpus


class ScriptureTextCorpus(DictionaryTextCorpus):
    @property
    @abstractmethod
    def versification(self) -> Versification:
        ...

    def get_text_sort_key(self, id: str) -> str:
        return get_scripture_text_sort_key(id)
