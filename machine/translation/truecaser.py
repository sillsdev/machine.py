from abc import ABC, abstractmethod
from typing import Sequence

from ..corpora.text_corpus import TextCorpus
from .trainer import Trainer


class Truecaser(ABC):
    @abstractmethod
    def create_trainer(self, corpus: TextCorpus) -> Trainer:
        ...

    @abstractmethod
    def train_segment(self, segment: Sequence[str], sentence_start: bool = True) -> None:
        ...

    @abstractmethod
    def truecase(self, segment: Sequence[str]) -> Sequence[str]:
        ...

    @abstractmethod
    def save(self) -> None:
        ...
