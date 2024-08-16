from abc import ABC, abstractmethod
from typing import Optional

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_engine import TranslationEngine
from ..translation.truecaser import Truecaser
from .thot.thot_model_factory import ThotModelFactory


class SmtModelFactory(ABC, ThotModelFactory):
    @abstractmethod
    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer: ...

    @abstractmethod
    def create_engine(
        self,
        tokenizer: Tokenizer[str, int, str],
        detokenizer: Detokenizer[str, str],
        truecaser: Optional[Truecaser] = None,
    ) -> TranslationEngine: ...

    @abstractmethod
    def create_truecaser_trainer(self, tokenizer: Tokenizer[str, int, str], target_corpus: TextCorpus) -> Trainer: ...

    @abstractmethod
    def create_truecaser(self) -> Truecaser: ...
