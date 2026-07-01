from abc import ABC, abstractmethod

from .word_alignment_matrix import WordAlignmentMatrix


class TransductiveWordAlignmentModel(ABC):

    @property
    @abstractmethod
    def training_alignment_count(self) -> int: ...

    @abstractmethod
    def get_training_alignment(self, n: int) -> WordAlignmentMatrix: ...
