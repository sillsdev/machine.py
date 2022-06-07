from __future__ import annotations

from abc import abstractmethod
from itertools import chain, islice
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple

from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .alignment_corpus import AlignmentCorpus
from .corpora_utils import get_split_indices
from .corpus import Corpus
from .parallel_text_corpus import ParallelTextCorpus
from .text import Text
from .text_row import TextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces


class TextCorpus(Corpus[TextRow]):
    @property
    @abstractmethod
    def texts(self) -> Iterable[Text]:
        ...

    def get_rows(self, text_ids: Optional[Iterable[str]] = None) -> ContextManagedGenerator[TextRow, None, None]:
        return ContextManagedGenerator(self._get_rows(text_ids))

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        text_id_set = set((t.id for t in self.texts) if text_ids is None else text_ids)
        for text in self.texts:
            if text.id in text_id_set:
                with text.get_rows() as rows:
                    yield from rows

    @property
    def missing_rows_allowed(self) -> bool:
        return any(t.missing_rows_allowed for t in self.texts)

    def count(self, include_empty: bool = True) -> int:
        return sum(t.count(include_empty) for t in self.texts)

    def tokenize(self, tokenizer: Tokenizer[str, int, str]) -> TextCorpus:
        def _tokenize(row: TextRow) -> TextRow:
            if len(row.segment) > 0:
                row.segment = list(tokenizer.tokenize(row.text))
            return row

        return self.transform(_tokenize)

    def detokenize(self, detokenizer: Detokenizer[str, str]) -> TextCorpus:
        def _detokenize(row: TextRow) -> TextRow:
            if len(row.segment) > 1:
                row.segment = [detokenizer.detokenize(row.segment)]
            return row

        return self.transform(_detokenize)

    def normalize(self, normalization_form: str) -> TextCorpus:
        def _normalize(row: TextRow) -> TextRow:
            row.segment = normalize(normalization_form, row.segment)
            return row

        return self.transform(_normalize)

    def nfc_normalize(self) -> TextCorpus:
        return self.normalize("NFC")

    def nfd_normalize(self) -> TextCorpus:
        return self.normalize("NFD")

    def nfkc_normalize(self) -> TextCorpus:
        return self.normalize("NFKC")

    def nfkd_normalize(self) -> TextCorpus:
        return self.normalize("NFKD")

    def lowercase(self) -> TextCorpus:
        def _lowercase(row: TextRow) -> TextRow:
            row.segment = lowercase(row.segment)
            return row

        return self.transform(_lowercase)

    def escape_spaces(self) -> TextCorpus:
        def _escape_spaces(row: TextRow) -> TextRow:
            row.segment = escape_spaces(row.segment)
            return row

        return self.transform(_escape_spaces)

    def unescape_spaces(self) -> TextCorpus:
        def _unescape_spaces(row: TextRow) -> TextRow:
            row.segment = unescape_spaces(row.segment)
            return row

        return self.transform(_unescape_spaces)

    def filter_texts(self, predicate: Callable[[Text], bool]) -> TextCorpus:
        return _TextFilterTextCorpus(self, predicate)

    def transform(self, transform: Callable[[TextRow], TextRow]) -> TextCorpus:
        return _TransformTextCorpus(self, transform)

    def align_rows(
        self,
        other: TextCorpus,
        alignment_corpus: Optional[AlignmentCorpus] = None,
        all_source_rows: bool = False,
        all_target_rows: bool = False,
    ) -> ParallelTextCorpus:
        from .standard_parallel_text_corpus import StandardParallelTextCorpus

        return StandardParallelTextCorpus(self, other, alignment_corpus, all_source_rows, all_target_rows)

    def filter_nonempty(self) -> TextCorpus:
        return self.filter(lambda r: not r.is_empty)

    def filter(self, predicate: Callable[[TextRow], bool]) -> TextCorpus:
        return self.filter_by_index(lambda r, _: predicate(r))

    def filter_by_index(self, predicate: Callable[[TextRow, int], bool]) -> TextCorpus:
        return _FilterTextCorpus(self, predicate)

    def take(self, count: int) -> TextCorpus:
        return _TakeTextCorpus(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple[TextCorpus, TextCorpus, int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda r, i: i not in split_indices and (include_empty or not r.is_empty))
        split_corpus = self.filter_by_index(lambda r, i: i in split_indices and (include_empty or not r.is_empty))

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)


def flatten_text_corpora(corpora: Iterable[TextCorpus]) -> TextCorpus:
    corpus_list = list(corpora)
    if len(corpus_list) == 1:
        return corpus_list[0]

    return _FlattenTextCorpus(corpus_list)


class _TransformTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, transform: Callable[[TextRow], TextRow]) -> None:
        self._corpus = corpus
        self._transform = transform

    @property
    def texts(self) -> Iterable[Text]:
        return self._corpus.texts

    @property
    def missing_rows_allowed(self) -> bool:
        return self._corpus.missing_rows_allowed

    def count(self, include_empty: bool = True) -> int:
        return self._corpus.count(include_empty)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from map(self._transform, rows)


class _TextFilterTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, predicate: Callable[[Text], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    @property
    def texts(self) -> Iterable[Text]:
        return (t for t in self._corpus.texts if self._predicate(t))

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows((t.id for t in self.texts) if text_ids is None else text_ids) as rows:
            yield from rows


class _FilterTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, predicate: Callable[[TextRow, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    @property
    def texts(self) -> Iterable[Text]:
        return self._corpus.texts

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeTextCorpus(TextCorpus):
    def __init__(self, corpus: TextCorpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    @property
    def texts(self) -> Iterable[Text]:
        return self._corpus.texts

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            yield from islice(rows, self._count)


class _FlattenTextCorpus(TextCorpus):
    def __init__(self, corpora: List[TextCorpus]) -> None:
        self._corpora = corpora

    @property
    def texts(self) -> Iterable[Text]:
        return chain.from_iterable(c.texts for c in self._corpora)

    @property
    def missing_rows_allowed(self) -> bool:
        return any(corpus.missing_rows_allowed for corpus in self._corpora)

    def count(self, include_empty: bool = True) -> int:
        return sum(corpus.count(include_empty) for corpus in self._corpora)

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[TextRow, None, None]:
        for corpus in self._corpora:
            with corpus.get_rows(text_ids) as rows:
                yield from rows
