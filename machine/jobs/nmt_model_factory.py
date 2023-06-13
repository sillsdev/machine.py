from abc import ABC, abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..translation.trainer import Trainer
from ..translation.translation_engine import TranslationEngine


class NmtModelFactory(ABC):
    @property
    @abstractmethod
    def train_tokenizer(self) -> bool:
        ...

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_model_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_engine(self) -> TranslationEngine:
        ...

    @abstractmethod
    def save_model(self) -> None:
        ...
