from abc import ABC, abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_model import TranslationModel


class NmtModelFactory(ABC):
    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def create_model(self) -> TranslationModel:
        ...

    @abstractmethod
    def create_model_trainer(
        self, source_language_tag: str, target_language_tag: str, corpus: ParallelTextCorpus
    ) -> Trainer:
        ...

    @abstractmethod
    def create_source_tokenizer(self) -> Tokenizer[str, int, str]:
        ...

    @abstractmethod
    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_detokenizer(self) -> Detokenizer[str, str]:
        ...

    @abstractmethod
    def save_model(self) -> None:
        ...
