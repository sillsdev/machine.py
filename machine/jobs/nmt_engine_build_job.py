import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence

from ..corpora.corpora_utils import batch
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .engine_build_job import EngineBuildJob
from .nmt_model_factory import NmtModelFactory
from .shared_file_service import DictToJsonWriter, PretranslationInfo, SharedFileService

logger = logging.getLogger(__name__)


class NmtEngineBuildJob(EngineBuildJob):
    def __init__(self, config: Any, nmt_model_factory: NmtModelFactory, shared_file_service: SharedFileService) -> None:
        self._nmt_model_factory = nmt_model_factory
        super().__init__(config, shared_file_service)

    def start_job(self) -> None:
        self._nmt_model_factory.init()

    def _get_progress_reporter(self, progress: Optional[Callable[[ProgressStatus], None]]) -> PhasedProgressReporter:
        if self._parallel_corpus_size > 0:
            phases = [
                Phase(message="Training NMT model", percentage=0.9),
                Phase(message="Pretranslating segments", percentage=0.1),
            ]
        else:
            phases = [Phase(message="Pretranslating segments", percentage=1.0)]
        return PhasedProgressReporter(progress, phases)

    def respond_to_no_training_corpus(self) -> None:
        logger.info("No matching entries in the source and target corpus - skipping training")

    def train_model(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        if check_canceled is not None:
            check_canceled()

        if self._nmt_model_factory.train_tokenizer:
            logger.info("Training source tokenizer")
            with self._nmt_model_factory.create_source_tokenizer_trainer(
                self._source_corpus
            ) as source_tokenizer_trainer:
                source_tokenizer_trainer.train(check_canceled=check_canceled)
                source_tokenizer_trainer.save()

            if check_canceled is not None:
                check_canceled()

            logger.info("Training target tokenizer")
            with self._nmt_model_factory.create_target_tokenizer_trainer(
                self._target_corpus
            ) as target_tokenizer_trainer:
                target_tokenizer_trainer.train(check_canceled=check_canceled)
                target_tokenizer_trainer.save()

            if check_canceled is not None:
                check_canceled()

        logger.info("Training NMT model")
        with progress_reporter.start_next_phase() as phase_progress, self._nmt_model_factory.create_model_trainer(
            self._parallel_corpus
        ) as model_trainer:
            model_trainer.train(progress=phase_progress, check_canceled=check_canceled)
            model_trainer.save()

    def batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
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
            batch_size = self._config["inference_batch_size"]
            for pi_batch in batch(src_pretranslations, batch_size):
                if check_canceled is not None:
                    check_canceled()
                _translate_batch(engine, pi_batch, writer)
                current_inference_step += len(pi_batch)
                phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))

    def save_model(self) -> None:
        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._nmt_model_factory.save_model()
            self._shared_file_service.save_model(
                model_path, f"models/{self._config.save_model + ''.join(model_path.suffixes)}"
            )


def _translate_batch(
    engine: TranslationEngine,
    batch: Sequence[PretranslationInfo],
    writer: DictToJsonWriter,
) -> None:
    source_segments = [pi["translation"] for pi in batch]
    for i, result in enumerate(engine.translate_batch(source_segments)):
        batch[i]["translation"] = result.translation
        writer.write(batch[i])  # type: ignore
