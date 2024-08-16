import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..tokenization.tokenizer_factory import create_tokenizer
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .word_alignment_file_service import WordAlignmentFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class WordAlignmentBuildJob:
    def __init__(
        self,
        config: Any,
        word_alignment_model_factory: WordAlignmentModelFactory,
        word_alignment_file_service: WordAlignmentFileService,
    ) -> None:
        self._word_alignment_model_factory = word_alignment_model_factory
        self._word_alignment_model_factory.init()
        self._config = config
        self._tokenizer = create_tokenizer(self._config.thot_align.tokenizer)
        self._word_alignment_file_service = word_alignment_file_service
        self._train_corpus_size = -1

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> int:
        if check_canceled is not None:
            check_canceled()

        progress_reporter = self._get_progress_reporter(progress)

        source_corpus = self._word_alignment_file_service.create_source_corpus()
        target_corpus = self._word_alignment_file_service.create_target_corpus()
        parallel_corpus: ParallelTextCorpus = source_corpus.align_rows(target_corpus)

        if parallel_corpus.count(include_empty=False) == 0:
            raise RuntimeError("No parallel corpus data found")

        train_corpus_size = self._train_model(parallel_corpus, progress_reporter, check_canceled)

        if check_canceled is not None:
            check_canceled()

        logger.info("Generating alignments")
        self._batch_inference(parallel_corpus, progress_reporter, check_canceled)

        self._save_model()
        return train_corpus_size

    def _get_progress_reporter(self, progress: Optional[Callable[[ProgressStatus], None]]) -> PhasedProgressReporter:
        phases = [
            Phase(message="Training Word Alignment model", percentage=0.9),
            Phase(message="Aligning segments", percentage=0.1),
        ]
        return PhasedProgressReporter(progress, phases)

    def _train_model(
        self,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> int:

        with progress_reporter.start_next_phase() as phase_progress, self._word_alignment_model_factory.create_model_trainer(
            self._tokenizer, parallel_corpus
        ) as trainer:
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()
            train_corpus_size = trainer.stats.train_corpus_size

        if check_canceled is not None:
            check_canceled()
        return train_corpus_size

    def _batch_inference(
        self,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        inference_step_count = parallel_corpus.count(include_empty=False)

        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            alignment_model = stack.enter_context(self._word_alignment_model_factory.create_alignment_model())
            writer = stack.enter_context(self._word_alignment_file_service.open_target_alignment_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["inference_batch_size"]
            segment_batch = list(parallel_corpus.lowercase().tokenize(self._tokenizer).take(batch_size))
            if check_canceled is not None:
                check_canceled()
            alignments = alignment_model.align_batch(segment_batch)
            if check_canceled is not None:
                check_canceled()

            def format_score(score: float) -> str:
                return f"{score:.8f}".rstrip("0").rstrip(".")

            for row, alignment in zip(parallel_corpus.get_rows(), alignments):
                source_segment = list(self._tokenizer.tokenize(row.source_text))
                target_segment = list(self._tokenizer.tokenize(row.target_text))
                word_pairs = alignment.to_aligned_word_pairs(include_null=True)
                alignment_model.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)

                word_alignment_info = {
                    "refs": [str(ref) for ref in row.source_refs],
                    "column_count": alignment.column_count,
                    "row_count": alignment.row_count,
                    "alignment": AlignedWordPair.to_string(word_pairs),
                }
                writer.write(word_alignment_info)

    def _save_model(self) -> None:
        logger.info("Saving model")
        model_path = self._word_alignment_model_factory.save_model()
        self._word_alignment_file_service.save_model(
            model_path, f"builds/{self._config['build_id']}/model{''.join(model_path.suffixes)}"
        )
