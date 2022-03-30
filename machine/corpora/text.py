from abc import abstractmethod

from .corpus import Corpus
from .text_row import TextRow


class Text(Corpus[TextRow]):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def sort_key(self) -> str:
        ...
