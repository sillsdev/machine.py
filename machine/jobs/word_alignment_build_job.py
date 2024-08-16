import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional

from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .engine_build_job import EngineBuildJob
from .shared_file_service import SharedFileService, WordAlignmentInfo
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class WordAlignmentBuildJob(EngineBuildJob):
    def __init__(
        self,
        config: Any,
        word_alignment_model_factory: WordAlignmentModelFactory,
        shared_file_service: SharedFileService,
    ) -> None:
        self._word_alignment_model_factory = word_alignment_model_factory
        super().__init__(config, shared_file_service)

    def start_job(self) -> None:
        self._word_alignment_model_factory.init()
        self._tokenizer = self._word_alignment_model_factory.create_tokenizer()
        logger.info(f"Tokenizer: {type(self._tokenizer).__name__}")

    def _get_progress_reporter(self, progress: Optional[Callable[[ProgressStatus], None]]) -> PhasedProgressReporter:
        phases = [
            Phase(message="Training Word Alignment model", percentage=0.9),
            Phase(message="Aligning segments", percentage=0.1),
        ]
        return PhasedProgressReporter(progress, phases)

    def respond_to_no_training_corpus(self) -> None:
        raise RuntimeError("No parallel corpus data found")

    def train_model(
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

    def batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        inference_step_count = self._parallel_corpus.count(include_empty=False)

        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            alignment_model = stack.enter_context(self._word_alignment_model_factory.create_alignment_model())
            self.init_corpus()
            writer = stack.enter_context(self._shared_file_service.open_target_alignment_writer())
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
                    )  # type: ignore
                )

    def save_model(self) -> None:
        logger.info("Saving model")
        model_path = self._word_alignment_model_factory.save_model()
        self._shared_file_service.save_model(
            model_path, f"builds/{self._config['build_id']}/model{''.join(model_path.suffixes)}"
        )
