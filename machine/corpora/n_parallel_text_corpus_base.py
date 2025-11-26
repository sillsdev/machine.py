from abc import ABC, abstractmethod
from typing import List, Sequence

from .corpus import Corpus
from .n_parallel_text_row import NParallelTextRow
from .text_corpus import TextCorpus


class NParallelTextCorpusBase(Corpus[NParallelTextRow], ABC):

    @property
    @abstractmethod
    def n(self) -> int: ...

    @property
    @abstractmethod
    def corpora(self) -> List[TextCorpus]: ...

    @abstractmethod
    def get_rows(self, text_ids: List[str]) -> Sequence[NParallelTextRow]: ...
