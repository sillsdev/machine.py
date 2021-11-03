from typing import Callable, Iterable

from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus import TextAlignmentCorpus


class FilteredTextAlignmentCorpus(TextAlignmentCorpus):
    def __init__(self, corpus: TextAlignmentCorpus, filter: Callable[[TextAlignmentCollection], bool]) -> None:
        self._corpus = corpus
        self._filter = filter

    @property
    def text_alignment_collections(self) -> Iterable[TextAlignmentCollection]:
        return (c for c in self._corpus.text_alignment_collections if self._filter(c))

    def __getitem__(self, id: str) -> TextAlignmentCollection:
        collection = self._corpus[id]
        if self._filter(collection):
            return collection
        return self.create_null_text_alignment_collection(id)

    def create_null_text_alignment_collection(self, id: str) -> TextAlignmentCollection:
        return self._corpus.create_null_text_alignment_collection(id)

    def invert(self) -> "FilteredTextAlignmentCorpus":
        return FilteredTextAlignmentCorpus(self._corpus.invert(), self._filter)
