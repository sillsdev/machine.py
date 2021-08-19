from abc import abstractmethod

from .ibm1_word_alignment_model import Ibm1WordAlignmentModel


class Ibm2WordAlignmentModel(Ibm1WordAlignmentModel):
    @abstractmethod
    def get_alignment_probability(
        self, source_length: int, source_index: int, target_length: int, target_index: int
    ) -> float:
        ...
