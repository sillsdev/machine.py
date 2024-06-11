from abc import ABC, abstractmethod
from pathlib import Path

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_engine import TranslationEngine
from ..translation.truecaser import Truecaser


class SmtModelFactory(ABC):
    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def create_tokenizer(self) -> Tokenizer[str, int, str]: ...

    @abstractmethod
    def create_detokenizer(self) -> Detokenizer[str, str]: ...

    @abstractmethod
    def create_model_trainer(self, tokenizer: Tokenizer[str, int, str], corpus: ParallelTextCorpus) -> Trainer: ...

    @abstractmethod
    def create_engine(
        self, tokenizer: Tokenizer[str, int, str], detokenizer: Detokenizer[str, str], truecaser: Truecaser
    ) -> TranslationEngine: ...

    @abstractmethod
    def create_truecaser_trainer(self, tokenizer: Tokenizer[str, int, str], target_corpus: TextCorpus) -> Trainer: ...

    @abstractmethod
    def create_truecaser(self) -> Truecaser: ...

    @abstractmethod
    def save_model(self) -> Path: ...
