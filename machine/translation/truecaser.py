from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..corpora.text_corpus import TextCorpus
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.whitespace_detokenizer import WHITESPACE_DETOKENIZER
from .trainer import Trainer
from .translation_result import TranslationResult


class Truecaser(ABC):
    @abstractmethod
    def create_trainer(self, corpus: TextCorpus) -> Trainer: ...

    @abstractmethod
    def train_segment(self, segment: Sequence[str], sentence_start: bool = True) -> None: ...

    @abstractmethod
    def truecase(self, segment: Sequence[str]) -> Sequence[str]: ...

    def truecase_translation_result(
        self, result: TranslationResult, detokenizer: Optional[Detokenizer] = None
    ) -> TranslationResult:
        if detokenizer is None:
            detokenizer = WHITESPACE_DETOKENIZER
        target_tokens = self.truecase(result.target_tokens)
        return TranslationResult(
            detokenizer.detokenize(target_tokens),
            result.source_tokens,
            target_tokens,
            result.confidences,
            result.sources,
            result.alignment,
            result.phrases,
        )

    @abstractmethod
    def save(self) -> None: ...
