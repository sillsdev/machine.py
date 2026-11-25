import logging
from typing import Any, Callable, Generator, Iterable, Optional, Tuple

from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..translation.translation_engine import TranslationEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_engine_build_job_base import NmtEngineBuildJobBase, join_surrogate_pairs_if_needed
from .nmt_model_factory import NmtModelFactory
from .translation_file_service import PretranslationInfo, TranslationFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class HuggingFaceNmtEngineBuildJob(NmtEngineBuildJobBase):
    def __init__(
        self,
        config: Any,
        nmt_model_factory: NmtModelFactory,
        translation_file_service: TranslationFileService,
        word_alignment_model_factory: WordAlignmentModelFactory,
    ) -> None:
        self._nmt_model_factory = nmt_model_factory
        self._nmt_model_factory.init()
        super().__init__(config, nmt_model_factory, translation_file_service, word_alignment_model_factory)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        if corpus_size > 0:
            if self._config.align_pretranslations:
                phases = [
                    Phase(message="Training NMT model", percentage=0.8, stage="train"),
                    Phase(message="Pretranslating segments", percentage=0.1, stage="inference"),
                    Phase(message="Aligning segments", percentage=0.1, stage="align"),
                ]
            else:
                phases = [
                    Phase(message="Training NMT model", percentage=0.9, stage="train"),
                    Phase(message="Pretranslating segments", percentage=0.1, stage="inference"),
                ]
        else:
            if self._config.align_pretranslations:
                phases = [
                    Phase(message="Pretranslating segments", percentage=0.9, stage="inference"),
                    Phase(message="Aligning segments", percentage=0.1, stage="align"),
                ]
            else:
                phases = [Phase(message="Pretranslating segments", percentage=1.0, stage="inference")]
        if self._config.get("post_edit_model"):
            phases = [Phase(p.message, p.percentage * 0.9, p.report_steps, p.stage) for p in phases]
            index = next(i for i, phase in enumerate(phases) if phase.stage == "inference") + 1
            phases.insert(index, Phase(message="Postediting pretranslations", percentage=0.1, stage="post_edit"))
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
        with (
            progress_reporter.start_next_phase() as phase_progress,
            self._nmt_model_factory.create_model_trainer(parallel_corpus) as model_trainer,
        ):
            model_trainer.train(progress=phase_progress, check_canceled=check_canceled)
            model_trainer.save()
            train_corpus_size = model_trainer.stats.train_corpus_size
        return train_corpus_size, float("nan")

    def _translate(
        self,
        engine: TranslationEngine,
        src_pretranslations: Iterable[PretranslationInfo],
        inference_step_count: int,
        phase_progress: Callable[[ProgressStatus], None],
        check_canceled: Optional[Callable[[], None]],
    ) -> Generator[Tuple[str, PretranslationInfo], None, None]:
        current_inference_step = 0
        phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
        batch_size: int = self._config["inference_batch_size"]
        for pt_batch in batch(src_pretranslations, batch_size):
            if check_canceled is not None:
                check_canceled()
            source_segments = [join_surrogate_pairs_if_needed(pt_info["translation"]) for pt_info in pt_batch]
            t_batch = engine.translate_batch(source_segments)
            for pt_info, result in zip(pt_batch, t_batch, strict=True):
                pt_info["translation"] = result.translation
                pt_info["sequenceConfidence"] = result.sequence_confidence
            current_inference_step += len(pt_batch)
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            yield from zip(source_segments, pt_batch)

    def _save_model(self) -> None:
        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._nmt_model_factory.save_model()
            self._translation_file_service.save_model(
                model_path, f"models/{self._config.save_model + ''.join(model_path.suffixes)}"
            )
