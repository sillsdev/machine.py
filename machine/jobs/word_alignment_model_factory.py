from abc import ABC, abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.word_alignment_model import WordAlignmentModel
from .thot.thot_model_factory import ThotModelFactory


class WordAlignmentModelFactory(ABC, ThotModelFactory):
    @abstractmethod
    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer: ...

    @abstractmethod
    def create_alignment_model(
        self,
    ) -> WordAlignmentModel: ...
