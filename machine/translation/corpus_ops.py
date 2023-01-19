from typing import Callable, Generator, Optional, Union

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.parallel_text_row import ParallelTextRow
from ..utils.progress_status import ProgressStatus
from .symmetrization_heuristic import SymmetrizationHeuristic
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
        from .thot import create_thot_symmetrized_word_alignment_model

        model = create_thot_symmetrized_word_alignment_model(aligner)
        model.heuristic = symmetrization_heuristic
        with model.create_trainer(corpus) as trainer:
            trainer.train(progress)
            trainer.save()
        aligner = model

    return _WordAlignParallelTextCorpus(corpus, aligner, batch_size)


def translate_corpus(
    corpus: ParallelTextCorpus, translation_engine: TranslationEngine, batch_size: int = 1024
) -> ParallelTextCorpus:
    return _TranslateParallelTextCorpus(corpus, translation_engine, batch_size)


class _WordAlignParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, aligner: WordAligner, batch_size: int) -> None:
        self._corpus = corpus
        self._aligner = aligner
        self._batch_size = batch_size

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.batch(self._batch_size) as batches:
            for batch in batches:
                alignments = self._aligner.align_batch(batch)
                for row, alignment in zip(batch, alignments):
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


class _TranslateParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, translation_engine: TranslationEngine, batch_size: int) -> None:
        self._corpus = corpus
        self._translation_engine = translation_engine
        self._batch_size = batch_size

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.batch(self._batch_size) as batches:
            for batch in batches:
                translations = self._translation_engine.translate_batch([r.source_segment for r in batch])
                for row, translation in zip(batch, translations):
                    row.target_segment = translation.target_segment
                    yield row
