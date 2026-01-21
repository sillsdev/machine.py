from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, Generator, Iterable, Optional

from .alignment_corpus import AlignmentCorpus
from .dictionary_alignment_corpus import DictionaryAlignmentCorpus
from .n_parallel_text_corpus import NParallelTextCorpus, default_row_ref_comparer
from .parallel_text_corpus import ParallelTextCorpus
from .parallel_text_row import ParallelTextRow
from .text_corpus import TextCorpus


class StandardParallelTextCorpus(ParallelTextCorpus):
    def __init__(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        alignment_corpus: Optional[AlignmentCorpus] = None,
        all_source_rows: bool = False,
        all_target_rows: bool = False,
        row_ref_comparer: Optional[Callable[[object, object], int]] = None,
    ) -> None:
        self._source_corpus = source_corpus
        self._target_corpus = target_corpus
        self._alignment_corpus = DictionaryAlignmentCorpus() if alignment_corpus is None else alignment_corpus
        self._all_source_rows = all_source_rows
        self._all_target_rows = all_target_rows
        self._n_parallel_text_corpus = NParallelTextCorpus([source_corpus, target_corpus])
        self._n_parallel_text_corpus.all_rows[:] = [self.all_source_rows, self.all_target_rows]
        self._row_ref_comparer = row_ref_comparer or default_row_ref_comparer

    @property
    def is_source_tokenized(self) -> bool:
        return self.source_corpus.is_tokenized

    @property
    def is_target_tokenized(self) -> bool:
        return self.target_corpus.is_tokenized

    @property
    def source_corpus(self) -> TextCorpus:
        return self._source_corpus

    @property
    def target_corpus(self) -> TextCorpus:
        return self._target_corpus

    @property
    def alignment_corpus(self) -> AlignmentCorpus:
        return self._alignment_corpus

    @property
    def all_source_rows(self) -> bool:
        return self._all_source_rows

    @property
    def all_target_rows(self) -> bool:
        return self._all_target_rows

    def _get_rows(self, text_ids: Optional[Iterable[str]]) -> Generator[ParallelTextRow, None, None]:
        with ExitStack() as stack:
            alignment_iterator = stack.enter_context(self.alignment_corpus.get_rows(text_ids))
            is_scripture = self._source_corpus.is_scripture and self._target_corpus.is_scripture
            for n_row in self._n_parallel_text_corpus.get_rows(text_ids):
                compare_alignment_corpus = -1
                alignment_row = next(alignment_iterator, None)
                if self._alignment_corpus is not None and all([len(n) > 0 for n in n_row.n_segments]):
                    while True:
                        if alignment_row is not None:
                            compare_alignment_corpus = self._row_ref_comparer(n_row.ref, alignment_row.ref)
                        else:
                            compare_alignment_corpus = 1
                        if compare_alignment_corpus >= 0:
                            break
                        alignment_row = next(alignment_iterator, None)
                yield ParallelTextRow(
                    n_row.text_id,
                    n_row.n_refs[0] if len(n_row.n_refs[0]) > 0 or not is_scripture else [n_row.ref],
                    n_row.n_refs[1] if len(n_row.n_refs[1]) > 0 or not is_scripture else [n_row.ref],
                    source_flags=n_row.n_flags[0],
                    target_flags=n_row.n_flags[1],
                    source_segment=n_row.n_segments[0],
                    target_segment=n_row.n_segments[1],
                    aligned_word_pairs=(
                        alignment_row.aligned_word_pairs
                        if compare_alignment_corpus == 0 and alignment_row is not None
                        else None
                    ),
                    data_type=n_row.data_type,
                )
