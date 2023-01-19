from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import ContextManager, Optional, Sequence, Type

from .translation_result import TranslationResult


class TranslationEngine(ContextManager["TranslationEngine"]):
    @abstractmethod
    def translate(self, segment: Sequence[str]) -> TranslationResult:
        ...

    @abstractmethod
    def translate_n(self, n: int, segment: Sequence[str]) -> Sequence[TranslationResult]:
        ...

    @abstractmethod
    def translate_batch(self, segments: Sequence[Sequence[str]]) -> Sequence[TranslationResult]:
        ...

    @abstractmethod
    def translate_n_batch(self, n: int, segments: Sequence[Sequence[str]]) -> Sequence[Sequence[TranslationResult]]:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> TranslationEngine:
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        self.close()
