from typing import Callable, Iterable, Optional

from .text import Text
from .text_corpus import TextCorpus


class FilteredTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, filter: Callable[[Text], bool]) -> None:
        self._corpus = corpus
        self._filter = filter

    @property
    def texts(self) -> Iterable[Text]:
        return (t for t in self._corpus.texts if self._filter(t))

    def get_text(self, id: str) -> Optional[Text]:
        text = self._corpus.get_text(id)
        if text is not None and self._filter(text):
            return text
        return None

    def get_text_sort_key(self, id: str) -> str:
        return self._corpus.get_text_sort_key(id)
