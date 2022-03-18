from typing import Generator, Iterable

from .corpora_helpers import gen
from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus_row import TextAlignmentCorpusRow


class MemoryTextAlignmentCollection(TextAlignmentCollection):
    def __init__(self, id: str, alignments: Iterable[TextAlignmentCorpusRow] = []) -> None:
        self._id = id
        self._alignments = list(alignments)

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._id

    def _get_rows(self) -> Generator[TextAlignmentCorpusRow, None, None]:
        return gen(self._alignments)
