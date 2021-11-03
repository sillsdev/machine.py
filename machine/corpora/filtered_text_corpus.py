from typing import Callable, Iterable

from .text import Text
from .text_corpus import TextCorpus


class FilteredTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, filter: Callable[[Text], bool]) -> None:
        self._corpus = corpus
        self._filter = filter

    @property
    def texts(self) -> Iterable[Text]:
        return (t for t in self._corpus.texts if self._filter(t))

    def __getitem__(self, id: str) -> Text:
        text = self._corpus[id]
        if self._filter(text):
            return text
        return self.create_null_text(id)
