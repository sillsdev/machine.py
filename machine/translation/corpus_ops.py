from typing import Callable, Generator, Iterable, Optional, Union

from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.parallel_text_row import ParallelTextRow
from ..utils.progress_status import ProgressStatus
from .symmetrization_heuristic import SymmetrizationHeuristic
from .transductive_word_alignment_model import TransductiveWordAlignmentModel
from .translation_engine import TranslationEngine
from .word_aligner import WordAligner
from .word_alignment_matrix import WordAlignmentMatrix
from .word_alignment_model import WordAlignmentModel


def word_align_corpus(
    corpus: ParallelTextCorpus,
    aligner: Union[WordAligner, int, str] = "fast_align",
    batch_size: int = 1024,
    symmetrization_heuristic: SymmetrizationHeuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND,
    progress: Optional[Callable[[ProgressStatus], None]] = None,
) -> ParallelTextCorpus:
    if isinstance(aligner, (int, str)):
        return _TrainedWordAlignParallelTextCorpus(corpus, aligner, symmetrization_heuristic, progress)

    if isinstance(aligner, TransductiveWordAlignmentModel):
        return _TransductiveWordAlignParallelTextCorpus(corpus, aligner)
    return _WordAlignParallelTextCorpus(corpus, aligner, batch_size)


def translate_corpus(
    corpus: ParallelTextCorpus, translation_engine: TranslationEngine, batch_size: int = 1024
) -> ParallelTextCorpus:
    return _TranslateParallelTextCorpus(corpus, translation_engine, batch_size)


class _WordAlignParallelTextCorpusBase(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus) -> None:
        self._corpus = corpus

    @property
    def is_source_tokenized(self) -> bool:
        return self._corpus.is_source_tokenized

    @property
    def is_target_tokenized(self) -> bool:
        return self._corpus.is_target_tokenized

    def count(self, include_empty: bool = True, text_ids: Optional[Iterable[str]] = None) -> int:
        # Aligning does not add or remove rows, so counting need not align, which may train a model.
        return self._corpus.count(include_empty, text_ids)


class _WordAlignParallelTextCorpus(_WordAlignParallelTextCorpusBase):
    def __init__(self, corpus: ParallelTextCorpus, aligner: WordAligner, batch_size: int) -> None:
        super().__init__(corpus)
        self._aligner = aligner
        self._batch_size = batch_size

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            for row_batch in batch(rows, self._batch_size):
                alignments = self._aligner.align_batch(row_batch)
                for row, alignment in zip(row_batch, alignments):
                    known_alignment = WordAlignmentMatrix.from_parallel_text_row(row)
                    if known_alignment is not None:
                        known_alignment.priority_symmetrize_with(alignment)
                        alignment = known_alignment
                    word_pairs = alignment.to_aligned_word_pairs()
                    if isinstance(self._aligner, WordAlignmentModel):
                        self._aligner.compute_aligned_word_pair_scores(
                            row.source_segment, row.target_segment, word_pairs
                        )
                    row.aligned_word_pairs = word_pairs
                    yield row


class _TransductiveWordAlignParallelTextCorpus(_WordAlignParallelTextCorpusBase):
    def __init__(self, corpus: ParallelTextCorpus, model: TransductiveWordAlignmentModel) -> None:
        super().__init__(corpus)
        self._model = model

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[ParallelTextRow, None, None]:
        yield from _get_transductive_rows(self._corpus, self._model, text_ids)


class _TrainedWordAlignParallelTextCorpus(_WordAlignParallelTextCorpusBase):
    def __init__(
        self,
        corpus: ParallelTextCorpus,
        aligner: Union[int, str],
        symmetrization_heuristic: SymmetrizationHeuristic,
        progress: Optional[Callable[[ProgressStatus], None]],
    ) -> None:
        super().__init__(corpus)
        self._aligner = aligner
        self._symmetrization_heuristic = symmetrization_heuristic
        self._progress = progress

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[ParallelTextRow, None, None]:
        from .thot import create_thot_symmetrized_word_alignment_model

        # Training on only the requested texts keeps the training-alignment index in sync with the rows.
        corpus = self._corpus.filter_texts(text_ids)
        # Training in the generator ties the model's lifetime to reading the rows, at the cost of
        # training a new model on each iteration.
        with create_thot_symmetrized_word_alignment_model(self._aligner) as model:
            model.heuristic = self._symmetrization_heuristic
            # Retain the alignments computed during training so that the corpus can be aligned
            # without a separate, potentially expensive, inference pass.
            model.emit_training_alignments = True
            with model.create_trainer(corpus) as trainer:
                trainer.train(self._progress)
                trainer.save()
            yield from _get_transductive_rows(corpus, model, None)


def _get_transductive_rows(
    corpus: ParallelTextCorpus, model: TransductiveWordAlignmentModel, text_ids: Optional[Iterable[str]]
) -> Generator[ParallelTextRow, None, None]:
    # The training alignments are keyed by the order in which the sentence pairs were added during
    # training, so the corpus the model was trained on must be iterated in full to keep the index in
    # sync; rows outside the requested texts are skipped rather than filtered out.
    text_id_set = None if text_ids is None else set(text_ids)
    with corpus.get_rows() as rows:
        for index, row in enumerate(rows):
            if text_id_set is not None and row.text_id not in text_id_set:
                continue
            alignment = model.get_training_alignment(index)
            known_alignment = WordAlignmentMatrix.from_parallel_text_row(row)
            if known_alignment is not None:
                known_alignment.priority_symmetrize_with(alignment)
                alignment = known_alignment
            word_pairs = alignment.to_aligned_word_pairs()
            if isinstance(model, WordAlignmentModel):
                model.compute_aligned_word_pair_scores(row.source_segment, row.target_segment, word_pairs)
            row.aligned_word_pairs = word_pairs
            yield row


class _TranslateParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, translation_engine: TranslationEngine, batch_size: int) -> None:
        self._corpus = corpus
        self._translation_engine = translation_engine
        self._batch_size = batch_size

    @property
    def is_source_tokenized(self) -> bool:
        return self._corpus.is_source_tokenized

    @property
    def is_target_tokenized(self) -> bool:
        return self._corpus.is_target_tokenized

    def _get_rows(self, text_ids: Optional[Iterable[str]] = None) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows(text_ids) as rows:
            for row_batch in batch(rows, self._batch_size):
                translations = self._translation_engine.translate_batch(
                    [r.source_segment if self.is_source_tokenized else r.source_text for r in row_batch]
                )
                for row, translation in zip(row_batch, translations):
                    row.target_segment = translation.target_tokens
                    yield row
