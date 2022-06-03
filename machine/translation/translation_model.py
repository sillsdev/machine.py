from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import ContextManager, Optional, Type

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from .trainer import Trainer
from .translation_engine import TranslationEngine


class TranslationModel(ContextManager["TranslationModel"]):
    @abstractmethod
    def create_engine(self) -> TranslationEngine:
        ...

    @abstractmethod
    def create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        ...

    def __enter__(self) -> TranslationModel:
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None
