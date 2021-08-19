from abc import abstractmethod
from typing import Optional, Union

from .word_alignment_model import WordAlignmentModel


class Ibm1WordAlignmentModel(WordAlignmentModel):
    @abstractmethod
    def get_translation_probability(
        self, source_word: Optional[Union[str, int]], target_word: Optional[Union[str, int]]
    ) -> float:
        ...
