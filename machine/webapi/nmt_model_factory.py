from abc import ABC, abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_model import TranslationModel


class NmtModelFactory(ABC):
    @abstractmethod
    def init(self, key: str) -> None:
        ...

    @abstractmethod
    def create_model(self, key: str) -> TranslationModel:
        ...

    @abstractmethod
    def create_model_trainer(
        self, key: str, source_language_tag: str, target_language_tag: str, corpus: ParallelTextCorpus
    ) -> Trainer:
        ...

    @abstractmethod
    def create_source_tokenizer(self, key: str) -> Tokenizer[str, int, str]:
        ...

    @abstractmethod
    def create_source_tokenizer_trainer(self, key: str, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_tokenizer_trainer(self, key: str, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_detokenizer(self, key: str) -> Detokenizer[str, str]:
        ...

    @abstractmethod
    def save_model(self, key: str) -> None:
        ...

    @abstractmethod
    def cleanup(self, key: str) -> None:
        ...
