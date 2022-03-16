import sys
from abc import abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Type

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.token_processors import NO_OP, TokenProcessor
from .trainer import Trainer
from .translation_engine import TranslationEngine


class TranslationModel(AbstractContextManager):
    @abstractmethod
    def create_engine(self) -> TranslationEngine:
        ...

    @abstractmethod
    def create_trainer(
        self,
        corpus: ParallelTextCorpus,
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
        max_corpus_count: int = sys.maxsize,
    ) -> Trainer:
        ...

    def __enter__(self) -> "TranslationModel":
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None
