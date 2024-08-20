import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Tuple

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..tokenization.tokenizer_factory import create_tokenizer
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .word_alignment_file_service import WordAlignmentFileService, WordAlignmentInfo
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
        self._config = config
        self._word_alignment_file_service = word_alignment_file_service
        self._train_corpus_size = -1

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, float]:
        if check_canceled is not None:
            check_canceled()

        self._start_job()
        self._init_corpus()
        progress_reporter = self._get_progress_reporter(progress)

        if self._parallel_corpus_size == 0:
            self._respond_to_no_training_corpus()
        else:
            self._train_model(progress_reporter, check_canceled)

        if check_canceled is not None:
            check_canceled()

        logger.info("Pretranslating segments")
        self._batch_inference(progress_reporter, check_canceled)

        self._save_model()
        return self._train_corpus_size, self._confidence

    def _init_corpus(self) -> None:
        logger.info("Downloading data files")
        if "_source_corpus" not in self.__dict__:
            self._source_corpus = self._word_alignment_file_service.create_source_corpus()
            self._target_corpus = self._word_alignment_file_service.create_target_corpus()
            self._parallel_corpus: ParallelTextCorpus = self._source_corpus.align_rows(self._target_corpus)
            self._parallel_corpus_size = self._parallel_corpus.count(include_empty=False)

    def _start_job(self) -> None:
        self._word_alignment_model_factory.init()
        self._tokenizer = create_tokenizer(self._config.thot.tokenizer)
        logger.info(f"Tokenizer: {type(self._tokenizer).__name__}")

    def _get_progress_reporter(self, progress: Optional[Callable[[ProgressStatus], None]]) -> PhasedProgressReporter:
        phases = [
            Phase(message="Training Word Alignment model", percentage=0.9),
            Phase(message="Aligning segments", percentage=0.1),
        ]
        return PhasedProgressReporter(progress, phases)

    def _respond_to_no_training_corpus(self) -> None:
        raise RuntimeError("No parallel corpus data found")

    def _train_model(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:

        with progress_reporter.start_next_phase() as phase_progress, self._word_alignment_model_factory.create_model_trainer(
            self._tokenizer, self._parallel_corpus
        ) as trainer:
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()
            self._train_corpus_size = trainer.stats.train_corpus_size
            self._confidence = -1

        if check_canceled is not None:
            check_canceled()

    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        inference_step_count = self._parallel_corpus.count(include_empty=False)

        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            alignment_model = stack.enter_context(self._word_alignment_model_factory.create_alignment_model())
            self._init_corpus()
            writer = stack.enter_context(self._word_alignment_file_service.open_target_alignment_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["inference_batch_size"]
            segment_batch = list(self._parallel_corpus.lowercase().tokenize(self._tokenizer).take(batch_size))
            if check_canceled is not None:
                check_canceled()
            alignments = alignment_model.align_batch(segment_batch)
            if check_canceled is not None:
                check_canceled()
            for row, alignment in zip(self._parallel_corpus.get_rows(), alignments):
                writer.write(
                    WordAlignmentInfo(
                        refs=[str(ref) for ref in row.source_refs],
                        column_count=alignment.column_count,
                        row_count=alignment.row_count,
                        alignment=str(alignment),
                    )
                )

    def _save_model(self) -> None:
        logger.info("Saving model")
        model_path = self._word_alignment_model_factory.save_model()
        self._word_alignment_file_service.save_model(
            model_path, f"builds/{self._config['build_id']}/model{''.join(model_path.suffixes)}"
        )
