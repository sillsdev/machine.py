import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence, Tuple

from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..tokenization.tokenizer_factory import create_detokenizer, create_tokenizer
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .shared_file_service_base import DictToJsonWriter
from .smt_model_factory import SmtModelFactory
from .translation_engine_build_job import TranslationEngineBuildJob
from .translation_file_service import PretranslationInfo, TranslationFileService

logger = logging.getLogger(__name__)


class SmtEngineBuildJob(TranslationEngineBuildJob):
    def __init__(
        self, config: Any, smt_model_factory: SmtModelFactory, shared_file_service: TranslationFileService
    ) -> None:
        self._smt_model_factory = smt_model_factory
        self._smt_model_factory.init()
        self._tokenizer = create_tokenizer(config.thot_mt.tokenizer)
        logger.info(f"Tokenizer: {type(self._tokenizer).__name__}")
        super().__init__(config, shared_file_service)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        phases = [
            Phase(message="Training SMT model", percentage=0.85),
            Phase(message="Training truecaser", percentage=0.05),
            Phase(message="Pretranslating segments", percentage=0.1),
        ]
        return PhasedProgressReporter(progress, phases)

    def _respond_to_no_training_corpus(self) -> Tuple[int, float]:
        raise RuntimeError("No parallel corpus data found")

    def _train_model(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Tuple[int, float]:

        with progress_reporter.start_next_phase() as phase_progress, self._smt_model_factory.create_model_trainer(
            self._tokenizer, parallel_corpus
        ) as trainer:
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()
            train_corpus_size = trainer.stats.train_corpus_size
            confidence = trainer.stats.metrics["bleu"] * 100

        with progress_reporter.start_next_phase() as phase_progress, self._smt_model_factory.create_truecaser_trainer(
            self._tokenizer, target_corpus
        ) as truecase_trainer:
            truecase_trainer.train(progress=phase_progress, check_canceled=check_canceled)
            truecase_trainer.save()

        if check_canceled is not None:
            check_canceled()
        return train_corpus_size, confidence

    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        with self._translation_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)

        with ExitStack() as stack:
            detokenizer = create_detokenizer(self._config.thot_mt.tokenizer)
            truecaser = self._smt_model_factory.create_truecaser()
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = stack.enter_context(self._smt_model_factory.create_engine(self._tokenizer, detokenizer, truecaser))
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
        logger.info("Saving model")
        model_path = self._smt_model_factory.save_model()
        self._translation_file_service.save_model(
            model_path, f"builds/{self._config['build_id']}/model{''.join(model_path.suffixes)}"
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
