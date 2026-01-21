from abc import abstractmethod

from .corpus import Corpus
from .data_type import DataType
from .text_row import TextRow


class Text(Corpus[TextRow]):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def sort_key(self) -> str: ...

    @property
    @abstractmethod
    def data_type(self) -> DataType: ...
