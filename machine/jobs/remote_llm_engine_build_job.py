import logging
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple

from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..translation.llm.llm_completion_client import LlmCompletionClient
from ..translation.llm.remote_llm_engine import RemoteLlmEngine
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_engine_build_job_base import NmtEngineBuildJobBase, join_surrogate_pairs_if_needed
from .nmt_model_factory import NmtModelFactory
from .translation_file_service import PretranslationInfo, TranslationFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UntranslatedVerse:
    text_id: str
    source_refs: Tuple[str, ...]


class RemoteLlmEngineBuildJob(NmtEngineBuildJobBase):

    def __init__(
        self,
        config: Any,
        nmt_model_factory: NmtModelFactory,
        translation_file_service: TranslationFileService,
        word_alignment_model_factory: WordAlignmentModelFactory,
        completion_client: LlmCompletionClient,
    ) -> None:
        self._completion_client = completion_client
        self.untranslated_verses: List[UntranslatedVerse] = []
        super().__init__(config, nmt_model_factory, translation_file_service, word_alignment_model_factory)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        phases = [Phase(message="Drafting pretranslations", percentage=0.9, stage="inference")]
        if self._config.get("align_pretranslations", False):
            phases.append(Phase(message="Aligning pretranslations", percentage=0.1, stage="align"))
        else:
            phases[0] = Phase(message="Drafting pretranslations", percentage=1.0, stage="inference")
        if self._config.get("post_edit_model"):
            phases = [Phase(p.message, p.percentage * 0.9, p.report_steps, p.stage) for p in phases]
            phases.insert(1, Phase(message="Postediting pretranslations", percentage=0.1, stage="post_edit"))
        return PhasedProgressReporter(progress, phases)

    def _respond_to_no_training_corpus(self) -> Tuple[int, float]:
        logger.info("No matching entries in the source and target corpus - drafting without examples")
        self._nmt_model_factory.init_example_retriever(ParallelTextCorpus.from_parallel_rows([]))
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
        self._nmt_model_factory.init_example_retriever(parallel_corpus)
        num_examples = parallel_corpus.count(False)
        logger.info("Loaded %d in-context examples", num_examples)
        return num_examples, float("nan")

    def _batch_inference(
        self,
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        self.untranslated_verses.clear()
        super()._batch_inference(parallel_training_corpus, progress_reporter, check_canceled)
        usage = self._completion_client.usage_summary
        logger.info(
            "Remote LLM usage: %d requests, %d prompt tokens, %d completion tokens, "
            "$%.6f reported cost (%d unpriced requests)",
            usage.request_count,
            usage.input_token_count,
            usage.output_token_count,
            usage.reported_cost,
            usage.unpriced_request_count,
        )
        if self.untranslated_verses:
            failed_verse_details = [f"{', '.join(verse.source_refs)}" for verse in self.untranslated_verses]
            logger.warning(
                "Unable to translate %d verses after retries: %s",
                len(self.untranslated_verses),
                "; ".join(failed_verse_details),
            )

    def _translate(
        self,
        engine: RemoteLlmEngine,
        src_pretranslations: Iterable[PretranslationInfo],
        inference_step_count: int,
        phase_progress: Callable[[ProgressStatus], None],
        check_canceled: Optional[Callable[[], None]],
    ) -> Generator[Tuple[str, PretranslationInfo], None, None]:
        current_step = 0
        phase_progress(ProgressStatus.from_step(current_step, inference_step_count))
        batch_size = self._config["inference_batch_size"]
        for pt_batch in batch(src_pretranslations, batch_size):
            if check_canceled is not None:
                check_canceled()
            source_segments = [join_surrogate_pairs_if_needed(pt_info["translation"]) for pt_info in pt_batch]
            results = engine.translate_batch(source_segments)
            for pt_info, result in zip(pt_batch, results, strict=True):
                pt_info["translation"] = result.translation
                pt_info["sequenceConfidence"] = result.sequence_confidence
                if result.error is not None:
                    self.untranslated_verses.append(UntranslatedVerse(pt_info["textId"], tuple(pt_info["sourceRefs"])))

            current_step += len(pt_batch)
            phase_progress(ProgressStatus.from_step(current_step, inference_step_count))
            yield from zip(source_segments, pt_batch, strict=True)

    def _save_model(self) -> None:
        return
