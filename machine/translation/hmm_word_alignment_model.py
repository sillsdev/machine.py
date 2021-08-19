from abc import abstractmethod

from .ibm1_word_alignment_model import Ibm1WordAlignmentModel


class HmmWordAlignmentModel(Ibm1WordAlignmentModel):
    @abstractmethod
    def get_alignment_probability(self, source_length: int, prev_source_index: int, source_index: int) -> float:
        ...
