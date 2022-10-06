from __future__ import annotations

from abc import abstractmethod

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from .trainer import Trainer
from .translation_engine import TranslationEngine


class TranslationModel(TranslationEngine):
    @abstractmethod
    def create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        ...

    def __enter__(self) -> TranslationModel:
        return self
