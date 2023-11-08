import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence

from ..corpora.corpora_utils import batch
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_model_factory import NmtModelFactory
from .shared_file_service import PretranslationInfo, PretranslationWriter, SharedFileService

logger = logging.getLogger(__name__)


class NmtEngineBuildJob:
    def __init__(self, config: Any, nmt_model_factory: NmtModelFactory, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._nmt_model_factory = nmt_model_factory
        self._shared_file_service = shared_file_service

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if check_canceled is not None:
            check_canceled()

        self._nmt_model_factory.init()

        logger.info("Downloading data files")
        source_corpus = self._shared_file_service.create_source_corpus()
        target_corpus = self._shared_file_service.create_target_corpus()
        parallel_corpus = source_corpus.align_rows(target_corpus)
        parallel_corpus_size = parallel_corpus.count(include_empty=False)

        if parallel_corpus_size > 0:
            phases = [
                Phase(message="Training NMT model", percentage=0.9),
                Phase(message="Pretranslating segments", percentage=0.1),
            ]
        else:
            phases = [Phase(message="Pretranslating segments", percentage=1.0)]
        progress_reporter = PhasedProgressReporter(progress, phases)

        if parallel_corpus_size > 0:
            if check_canceled is not None:
                check_canceled()

            if self._nmt_model_factory.train_tokenizer:
                logger.info("Training source tokenizer")
                with self._nmt_model_factory.create_source_tokenizer_trainer(source_corpus) as source_tokenizer_trainer:
                    source_tokenizer_trainer.train(check_canceled=check_canceled)
                    source_tokenizer_trainer.save()

                if check_canceled is not None:
                    check_canceled()

                logger.info("Training target tokenizer")
                with self._nmt_model_factory.create_target_tokenizer_trainer(target_corpus) as target_tokenizer_trainer:
                    target_tokenizer_trainer.train(check_canceled=check_canceled)
                    target_tokenizer_trainer.save()

                if check_canceled is not None:
                    check_canceled()

            logger.info("Training NMT model")
            with progress_reporter.start_next_phase() as phase_progress, self._nmt_model_factory.create_model_trainer(
                parallel_corpus
            ) as model_trainer:
                model_trainer.train(progress=phase_progress, check_canceled=check_canceled)
                model_trainer.save()
        else:
            logger.info("No matching entries in the source and target corpus - skipping training")

        if check_canceled is not None:
            check_canceled()

        logger.info("Pretranslating segments")
        with self._shared_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)
        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = stack.enter_context(self._nmt_model_factory.create_engine())
            src_pretranslations = stack.enter_context(self._shared_file_service.get_source_pretranslations())
            writer = stack.enter_context(self._shared_file_service.open_target_pretranslation_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["batch_size"]
            translate_batch = TranslateBatch(batch_size)
            for pi_batch in batch(src_pretranslations, batch_size):
                if check_canceled is not None:
                    check_canceled()
                translate_batch.translate(engine, pi_batch, writer)
                current_inference_step += len(pi_batch)
                phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))


batch_divisor = 1


class TranslateBatch:
    def __init__(self, initial_batch_size):
        self.batch_size = initial_batch_size

    def translate(
        self,
        engine: TranslationEngine,
        batch: Sequence[PretranslationInfo],
        writer: PretranslationWriter,
    ) -> None:
        while True:
            source_segments = [pi["translation"] for pi in batch]
            outer_batch_size = len(source_segments)
            try:
                for step in range(0, outer_batch_size, self.batch_size):
                    for i, result in enumerate(engine.translate_batch(source_segments[step : step + self.batch_size])):
                        batch[i + step]["translation"] = result.translation
                for i in range(len(source_segments)):
                    writer.write(batch[i])
                break
            except Exception:
                self.batch_size = max(self.batch_size // 2, 1)
                logger.info(f"Out of memory error, reducing batch size to {self.batch_size}")
