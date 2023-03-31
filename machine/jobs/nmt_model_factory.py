from abc import ABC, abstractmethod
from typing import Optional

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
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
    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Optional[Trainer]:
        ...

    @abstractmethod
    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Optional[Trainer]:
        ...

    @abstractmethod
    def create_model_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        ...

    @abstractmethod
    def save_model(self) -> None:
        ...
