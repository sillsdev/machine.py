from abc import abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Iterable, List, Optional, Sequence, Type, Union, overload

from .translation_result import TranslationResult


class TranslationEngine(AbstractContextManager):
    @overload
    @abstractmethod
    def translate(self, segment: Sequence[str]) -> TranslationResult:
        ...

    @overload
    @abstractmethod
    def translate(self, segment: Sequence[str], n: int) -> List[TranslationResult]:
        ...

    @abstractmethod
    def translate(
        self, segment: Sequence[str], n: Optional[int] = None
    ) -> Union[TranslationResult, List[TranslationResult]]:
        ...

    @overload
    @abstractmethod
    def translate_batch(self, segments: Iterable[Sequence[str]]) -> Iterable[TranslationResult]:
        ...

    @overload
    @abstractmethod
    def translate_batch(self, segments: Iterable[Sequence[str]], n: int) -> Iterable[List[TranslationResult]]:
        ...

    @abstractmethod
    def translate_batch(
        self, segments: Iterable[Sequence[str]], n: Optional[int] = None
    ) -> Union[Iterable[TranslationResult], Iterable[List[TranslationResult]]]:
        ...

    def __enter__(self) -> "TranslationEngine":
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None
