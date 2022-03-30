from abc import abstractmethod

from .alignment_row import AlignmentRow
from .corpus import Corpus


class AlignmentCollection(Corpus[AlignmentRow]):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def sort_key(self) -> str:
        ...
