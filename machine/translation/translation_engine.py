from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import ContextManager, Iterable, List, Optional, Sequence, Type

from .translation_result import TranslationResult


class TranslationEngine(ContextManager["TranslationEngine"]):
    @abstractmethod
    def translate(self, segment: Sequence[str]) -> TranslationResult:
        ...

    @abstractmethod
    def translate_n(self, n: int, segment: Sequence[str]) -> List[TranslationResult]:
        ...

    @abstractmethod
    def translate_batch(
        self, segments: Iterable[Sequence[str]], batch_size: Optional[int] = None
    ) -> Iterable[TranslationResult]:
        ...

    @abstractmethod
    def translate_n_batch(
        self, n: int, segments: Iterable[Sequence[str]], batch_size: Optional[int] = None
    ) -> Iterable[List[TranslationResult]]:
        ...

    def __enter__(self) -> TranslationEngine:
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None
