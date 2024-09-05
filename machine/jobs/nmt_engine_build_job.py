import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence, Tuple

from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_model_factory import NmtModelFactory
from .shared_file_service_base import DictToJsonWriter
from .translation_engine_build_job import TranslationEngineBuildJob
from .translation_file_service import PretranslationInfo, TranslationFileService

logger = logging.getLogger(__name__)


class NmtEngineBuildJob(TranslationEngineBuildJob):
    def __init__(
        self, config: Any, nmt_model_factory: NmtModelFactory, translation_file_service: TranslationFileService
    ) -> None:
        self._nmt_model_factory = nmt_model_factory
        self._nmt_model_factory.init()
        super().__init__(config, translation_file_service)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        if corpus_size > 0:
            phases = [
                Phase(message="Training NMT model", percentage=0.9),
                Phase(message="Pretranslating segments", percentage=0.1),
            ]
        else:
            phases = [Phase(message="Pretranslating segments", percentage=1.0)]
        return PhasedProgressReporter(progress, phases)

    def _respond_to_no_training_corpus(self) -> Tuple[int, float]:
        logger.info("No matching entries in the source and target corpus - skipping training")
        return 0, float("nan")

    def _train_model(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Tuple[int, float]:
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
            train_corpus_size = model_trainer.stats.train_corpus_size
        return train_corpus_size, float("nan")

    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        logger.info("Pretranslating segments")
        with self._translation_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)
        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = stack.enter_context(self._nmt_model_factory.create_engine())
            src_pretranslations = stack.enter_context(self._translation_file_service.get_source_pretranslations())
            writer = stack.enter_context(self._translation_file_service.open_target_pretranslation_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["inference_batch_size"]
            for pi_batch in batch(src_pretranslations, batch_size):
                if check_canceled is not None:
                    check_canceled()
                _translate_batch(engine, pi_batch, writer)
                current_inference_step += len(pi_batch)
                phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))

    def _save_model(self) -> None:
        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._nmt_model_factory.save_model()
            self._translation_file_service.save_model(
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
        writer.write(batch[i])
