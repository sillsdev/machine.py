from abc import abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Iterable, List, Optional, Sequence, Type

from .translation_result import TranslationResult


class TranslationEngine(AbstractContextManager):
    @abstractmethod
    def translate(self, segment: Sequence[str]) -> TranslationResult:
        ...

    @abstractmethod
    def translate_n(self, n: int, segment: Sequence[str]) -> List[TranslationResult]:
        ...

    def translate_many(self, segments: Iterable[Sequence[str]]) -> Iterable[TranslationResult]:
        for segment in segments:
            yield self.translate(segment)

    def translate_many_n(self, n: int, segments: Iterable[Sequence[str]]) -> Iterable[List[TranslationResult]]:
        for segment in segments:
            yield self.translate_n(n, segment)

    def __enter__(self) -> "TranslationEngine":
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None
