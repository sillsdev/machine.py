from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from .text_corpus import TextCorpus
from .n_parallel_text_row import NParallelTextRow
from .corpus import Corpus


class NParallelTextCorpusBase(Corpus[NParallelTextRow], ABC):

    @property
    @abstractmethod
    def n(self) -> int: ...

    @property
    @abstractmethod
    def corpora(self) -> List[TextCorpus]: ...

    @abstractmethod
    def get_rows(self, text_ids: List[str]) -> Sequence[NParallelTextRow]: ...
