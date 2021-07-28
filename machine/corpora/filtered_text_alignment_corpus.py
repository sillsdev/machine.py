from typing import Callable, Iterable, Optional

from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus import TextAlignmentCorpus


class FilteredTextAlignmentCorpus(TextAlignmentCorpus):
    def __init__(self, corpus: TextAlignmentCorpus, filter: Callable[[TextAlignmentCollection], bool]) -> None:
        self._corpus = corpus
        self._filter = filter

    @property
    def text_alignment_collections(self) -> Iterable[TextAlignmentCollection]:
        return (c for c in self._corpus.text_alignment_collections if self._filter(c))

    def get_text_alignment_collection(self, id: str) -> Optional[TextAlignmentCollection]:
        collection = self._corpus.get_text_alignment_collection(id)
        if collection is not None and self._filter(collection):
            return collection
        return None

    def invert(self) -> "FilteredTextAlignmentCorpus":
        return FilteredTextAlignmentCorpus(self._corpus.invert(), self._filter)

    def get_text_alignment_collection_sort_key(self, id: str) -> str:
        return self._corpus.get_text_alignment_collection_sort_key(id)
