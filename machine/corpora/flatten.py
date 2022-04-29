from itertools import chain
from typing import Generator, Iterable, List, Optional, cast, overload

from .alignment_collection import AlignmentCollection
from .alignment_corpus import AlignmentCorpus
from .alignment_row import AlignmentRow
from .corpus import Corpus
from .parallel_text_corpus import ParallelTextCorpus
from .parallel_text_row import ParallelTextRow
from .text import Text
from .text_corpus import TextCorpus
from .text_row import TextRow


@overload
def flatten(corpora: Iterable[TextCorpus]) -> TextCorpus:
    ...


@overload
def flatten(corpora: Iterable[AlignmentCorpus]) -> AlignmentCorpus:
    ...


@overload
def flatten(corpora: Iterable[ParallelTextCorpus]) -> ParallelTextCorpus:
    ...


def flatten(corpora: Iterable[Corpus]) -> Corpus:
    corpus_list = list(corpora)
    if len(corpus_list) == 0:
        raise ValueError("No corpora specified.")

    if len(corpus_list) == 1:
        return corpus_list[0]

    if any(type(corpus_list[0]) != type(corpus) for corpus in corpus_list[1:]):
        raise TypeError("All corpora must be of the same type.")

    if isinstance(corpus_list[0], TextCorpus):
        return _FlattenTextCorpus(cast(List[TextCorpus], corpus_list))
    if isinstance(corpus_list[0], AlignmentCorpus):
        return _FlattenAlignmentCorpus(cast(List[AlignmentCorpus], corpus_list))
    return _FlattenParallelTextCorpus(cast(List[ParallelTextCorpus], corpus_list))


class _FlattenTextCorpus(TextCorpus):
    def __init__(self, corpora: List[TextCorpus]) -> None:
        self._corpora = corpora

    @property
    def texts(self) -> Iterable[Text]:
        return chain.from_iterable(c.texts for c in self._corpora)

    @property
    def missing_rows_allowed(self) -> bool:
        return any(c.missing_rows_allowed for c in self._corpora)

    def count(self, include_empty: bool = True) -> int:
        return sum(c.count(include_empty) for c in self._corpora)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        for corpus in self._corpora:
            with corpus.get_rows(text_ids) as rows:
                yield from rows


class _FlattenAlignmentCorpus(AlignmentCorpus):
    def __init__(self, corpora: List[AlignmentCorpus]) -> None:
        self._corpora = corpora

    @property
    def alignment_collections(self) -> Iterable[AlignmentCollection]:
        return chain.from_iterable(c.alignment_collections for c in self._corpora)

    @property
    def missing_rows_allowed(self) -> bool:
        return any(c.missing_rows_allowed for c in self._corpora)

    def count(self, include_empty: bool = True) -> int:
        return sum(c.count(include_empty) for c in self._corpora)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[AlignmentRow, None, None]:
        for corpus in self._corpora:
            with corpus.get_rows(text_ids) as rows:
                yield from rows


class _FlattenParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpora: List[ParallelTextCorpus]) -> None:
        self._corpora = corpora

    @property
    def missing_rows_allowed(self) -> bool:
        return any(c.missing_rows_allowed for c in self._corpora)

    def count(self, include_empty: bool = True) -> int:
        return sum(c.count(include_empty) for c in self._corpora)

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        for corpus in self._corpora:
            with corpus.get_rows() as rows:
                yield from rows
