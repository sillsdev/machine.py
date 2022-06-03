from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import ContextManager, Generator, Iterable, List, Optional, Sequence, Type, Union, overload

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.parallel_text_row import ParallelTextRow
from .translation_result import TranslationResult


class TranslationEngine(ContextManager["TranslationEngine"]):
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

    def __enter__(self) -> TranslationEngine:
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return None


def translate_corpus(
    translation_engine: TranslationEngine, corpus: ParallelTextCorpus, buffer_size: int = 1024
) -> ParallelTextCorpus:
    return _TranslateParallelTextCorpus(corpus, translation_engine, buffer_size)


class _TranslateParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, translation_engine: TranslationEngine, buffer_size: int) -> None:
        self._corpus = corpus
        self._translation_engine = translation_engine
        self._buffer_size = buffer_size

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        buffer: List[ParallelTextRow] = []
        with self._corpus.get_rows() as rows:
            for row in rows:
                buffer.append(row)
                if len(buffer) == self._buffer_size:
                    self._translate(buffer)
                    yield from buffer
                    buffer.clear()
            if len(buffer) > 0:
                self._translate(buffer)
                yield from buffer

    def _translate(self, buffer: List[ParallelTextRow]) -> None:
        translations = self._translation_engine.translate_batch(r.source_segment for r in buffer)
        for row, translation in zip(buffer, translations):
            row.target_segment = translation.target_segment
