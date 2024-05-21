import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence, Tuple

from ..corpora.corpora_utils import batch
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .shared_file_service import PretranslationInfo, PretranslationWriter, SharedFileService
from .smt_model_factory import SmtModelFactory

logger = logging.getLogger(__name__)


class SmtEngineBuildJob:
    def __init__(self, config: Any, smt_model_factory: SmtModelFactory, shared_file_service: SharedFileService) -> None:
        self._config = config
        self._smt_model_factory = smt_model_factory
        self._shared_file_service = shared_file_service

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, float]:
        if check_canceled is not None:
            check_canceled()

        self._smt_model_factory.init()
        tokenizer = self._smt_model_factory.create_tokenizer()
        logger.info(f"Tokenizer: {type(tokenizer).__name__}")

        logger.info("Downloading data files")
        source_corpus = self._shared_file_service.create_source_corpus()
        target_corpus = self._shared_file_service.create_target_corpus()
        parallel_corpus = source_corpus.align_rows(target_corpus)
        parallel_corpus_size = parallel_corpus.count(include_empty=False)
        if parallel_corpus_size == 0:
            raise RuntimeError("No parallel corpus data found")

        with self._shared_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)
        phases = [
            Phase(message="Training SMT model", percentage=0.85),
            Phase(message="Training truecaser", percentage=0.05),
            Phase(message="Pretranslating segments", percentage=0.1),
        ]
        progress_reporter = PhasedProgressReporter(progress, phases)

        if check_canceled is not None:
            check_canceled()

        with progress_reporter.start_next_phase() as phase_progress, self._smt_model_factory.create_model_trainer(
            tokenizer, parallel_corpus
        ) as trainer:
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()
            train_corpus_size = trainer.stats.train_corpus_size
            confidence = trainer.stats.metrics["bleu"] * 100

        with progress_reporter.start_next_phase() as phase_progress, self._smt_model_factory.create_truecaser_trainer(
            tokenizer, target_corpus
        ) as truecase_trainer:
            truecase_trainer.train(progress=phase_progress, check_canceled=check_canceled)
            truecase_trainer.save()

        if check_canceled is not None:
            check_canceled()

        with ExitStack() as stack:
            detokenizer = self._smt_model_factory.create_detokenizer()
            truecaser = self._smt_model_factory.create_truecaser()
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = stack.enter_context(self._smt_model_factory.create_engine(tokenizer, detokenizer, truecaser))
            src_pretranslations = stack.enter_context(self._shared_file_service.get_source_pretranslations())
            writer = stack.enter_context(self._shared_file_service.open_target_pretranslation_writer())
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["pretranslation_batch_size"]
            for pi_batch in batch(src_pretranslations, batch_size):
                if check_canceled is not None:
                    check_canceled()
                _translate_batch(engine, pi_batch, writer)
                current_inference_step += len(pi_batch)
                phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))

        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._smt_model_factory.save_model()
            self._shared_file_service.save_model(model_path, self._config.save_model + "".join(model_path.suffixes))
        return train_corpus_size, confidence


def _translate_batch(
    engine: TranslationEngine,
    batch: Sequence[PretranslationInfo],
    writer: PretranslationWriter,
) -> None:
    source_segments = [pi["translation"] for pi in batch]
    for i, result in enumerate(engine.translate_batch(source_segments)):
        batch[i]["translation"] = result.translation
        writer.write(batch[i])
