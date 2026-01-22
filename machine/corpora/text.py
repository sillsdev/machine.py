from abc import abstractmethod

from .corpus import Corpus
from .text_row import TextRow
from .text_row_content_type import TextRowContentType


class Text(Corpus[TextRow]):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def sort_key(self) -> str: ...
