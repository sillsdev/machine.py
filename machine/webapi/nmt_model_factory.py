from abc import ABC, abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..translation.trainer import Trainer
from ..translation.translation_model import TranslationModel


class NmtModelFactory(ABC):
    @abstractmethod
    def init(self, engine_id: str) -> None:
        ...

    @abstractmethod
    def create_model(self, engine_id: str) -> TranslationModel:
        ...

    @abstractmethod
    def create_model_trainer(self, engine_id: str, corpus: ParallelTextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_source_tokenizer(self, engine_id: str) -> Tokenizer[str, int, str]:
        ...

    @abstractmethod
    def create_source_tokenizer_trainer(self, engine_id: str, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_tokenizer_trainer(self, engine_id: str, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def create_target_detokenizer(self, engine_id: str) -> Detokenizer[str, str]:
        ...

    @abstractmethod
    def cleanup(self, engine_id: str) -> None:
        ...
