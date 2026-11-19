import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpora_utils import batch
from ..corpora.flatten import flatten
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..corpora.token_processors import lowercase
from ..tokenization.tokenizer_factory import create_tokenizer
from ..translation.llm.example import Example
from ..translation.llm.example_retriever import ExampleRetriever
from ..translation.llm.example_retriever_factory import ExampleRetrieverFactory
from ..translation.llm.example_retriever_method import ExampleRetrievalMethod
from ..translation.llm.llm_completion_client import LlmCompletionClient, LlmUsageTotals
from ..translation.llm.remote_llm_engine import RemoteLlmEngine
from ..translation.transductive_word_alignment_model import TransductiveWordAlignmentModel
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .translation_engine_build_job import TranslationEngineBuildJob
from .translation_file_service import PretranslationInfo, TranslationFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UntranslatedVerse:
    text_id: str
    source_refs: Tuple[str, ...]


class RemoteLlmEngineBuildJob(TranslationEngineBuildJob):

    def __init__(
        self,
        config: Any,
        completion_client: LlmCompletionClient,
        translation_file_service: TranslationFileService,
        word_alignment_model_factory: WordAlignmentModelFactory,
        retrieval_method: ExampleRetrievalMethod = ExampleRetrievalMethod.TFIDF,
        num_examples: int = 100,
        batch_size: int = 15,
        num_workers: int = 4,
    ) -> None:
        if num_examples < 0:
            raise ValueError("num_examples must be non-negative.")
        self._completion_client = completion_client
        self._retrieval_method = retrieval_method
        self._num_examples = num_examples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.usage_totals = LlmUsageTotals()
        self.untranslated_verses: List[UntranslatedVerse] = []
        self._example_retriever: Optional[ExampleRetriever] = None
        self._alignment_model_factory = word_alignment_model_factory
        self._alignment_model_factory.init()
        super().__init__(config, translation_file_service)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        phases = [Phase(message="Drafting pretranslations", percentage=0.9, stage="inference")]
        if self._config.get("align_pretranslations", False):
            phases.append(Phase(message="Aligning pretranslations", percentage=0.1, stage="align"))
        else:
            phases[0] = Phase(message="Drafting pretranslations", percentage=1.0, stage="inference")
        return PhasedProgressReporter(progress, phases)

    def _respond_to_no_training_corpus(self) -> Tuple[int, float]:
        logger.info("No matching entries in the source and target corpus - drafting without examples")
        self._example_retriever = ExampleRetrieverFactory.create_example_retriever(self._retrieval_method, [])
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
        examples: List[Example] = []
        with parallel_corpus.get_rows() as rows:
            for row in rows:
                if not row.is_empty:
                    examples.append(Example(row.source_text, row.target_text))
        self._example_retriever = ExampleRetrieverFactory.create_example_retriever(self._retrieval_method, examples)
        logger.info("Loaded %d in-context examples", len(examples))
        return len(examples), float("nan")

    def _batch_inference(
        self,
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        self.untranslated_verses.clear()
        if self._example_retriever is None:
            self._example_retriever = ExampleRetrieverFactory.create_example_retriever(self._retrieval_method, [])
        with self._translation_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)

        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = RemoteLlmEngine(
                self._completion_client,
                self._example_retriever,
                self._config.src_lang,
                self._config.trg_lang,
                self._num_examples,
                self._batch_size,
                self._num_workers,
                self.usage_totals,
            )
            stack.enter_context(engine)
            src_pretranslations = stack.enter_context(self._translation_file_service.get_source_pretranslations())
            pretranslations = self._translate(
                engine, src_pretranslations, inference_step_count, phase_progress, check_canceled
            )
            if self._config.get("align_pretranslations", False):
                if self._alignment_model_factory is None:
                    raise ValueError(
                        "A word alignment model factory is required when align_pretranslations is enabled."
                    )
                results: Iterable[PretranslationInfo] = self._align(
                    pretranslations, parallel_training_corpus, progress_reporter, check_canceled
                )
            else:
                results = (pt_info for _, pt_info in pretranslations)
            writer = stack.enter_context(self._translation_file_service.open_target_pretranslation_writer())
            for pt_info in results:
                writer.write(pt_info)
        usage = self.usage_totals.summary
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
            source_segments = [pt_info["translation"] for pt_info in pt_batch]
            results = engine.translate_batch(source_segments)
            failed_indices = set(engine.failed_segment_indices)
            for index, (pt_info, result) in enumerate(zip(pt_batch, results, strict=True)):
                pt_info["translation"] = result.translation
                pt_info["sequenceConfidence"] = result.sequence_confidence
                if index in failed_indices:
                    self.untranslated_verses.append(UntranslatedVerse(pt_info["textId"], tuple(pt_info["sourceRefs"])))

            current_step += len(pt_batch)
            phase_progress(ProgressStatus.from_step(current_step, inference_step_count))
            yield from zip(source_segments, pt_batch, strict=True)

    def _align(
        self,
        pretranslations: Iterable[Tuple[str, PretranslationInfo]],
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Generator[PretranslationInfo, None, None]:
        if check_canceled is not None:
            check_canceled()

        tokenizer = create_tokenizer(self._config.thot_align.tokenizer)
        with TemporaryDirectory() as temp_dir:
            # Spool the translated pretranslations to disk so that the aligner can make multiple
            # passes over them without keeping the entire corpus in memory.
            logger.info("Aligning source to pretranslations")
            source_path = Path(temp_dir) / "pretranslations.src.txt"
            translation_path = Path(temp_dir) / "pretranslations.trg.txt"
            pretranslations_path = Path(temp_dir) / "pretranslations.json"
            with (
                source_path.open("w", encoding="utf-8", newline="\n") as source_file,
                translation_path.open("w", encoding="utf-8", newline="\n") as translation_file,
                pretranslations_path.open("w", encoding="utf-8", newline="\n") as pretranslations_file,
            ):
                for source_segment, pt_info in pretranslations:
                    source_file.write(source_segment + "\n")
                    translation_file.write(pt_info["translation"] + "\n")
                    pretranslations_file.write(json.dumps(pt_info, ensure_ascii=False) + "\n")

            if check_canceled is not None:
                check_canceled()

            parallel_pretranslation_corpus = TextFileTextCorpus(source_path).align_rows(
                TextFileTextCorpus(translation_path)
            )

            # The pretranslations are placed at the beginning of the training corpus so that a
            # transductive model's training alignments can be matched back to them by index.
            alignment_parallel_corpus = flatten([parallel_pretranslation_corpus, parallel_training_corpus])

            logger.info("Training aligner")
            with (
                progress_reporter.start_next_phase() as phase_progress,
                self._alignment_model_factory.create_model_trainer(tokenizer, alignment_parallel_corpus) as trainer,
            ):
                trainer.train(progress=phase_progress, check_canceled=check_canceled)
                trainer.save()

            if check_canceled is not None:
                check_canceled()

            logger.info("Aligning pretranslations")
            batch_size = self._config["inference_batch_size"]
            with (
                self._alignment_model_factory.create_alignment_model() as alignment_model,
                parallel_pretranslation_corpus.tokenize(tokenizer).get_rows() as rows,
            ):
                transductive_model: Optional[TransductiveWordAlignmentModel] = None
                if isinstance(alignment_model, TransductiveWordAlignmentModel):
                    transductive_model = alignment_model
                index = 0
                rows_with_pretranslations = zip(_read_pretranslations(pretranslations_path), rows, strict=True)
                for pt_batch in batch(rows_with_pretranslations, batch_size):
                    if check_canceled is not None:
                        check_canceled()
                        # The aligner is trained on lowercased tokens, so it must also be given
                    # lowercased tokens when aligning and scoring; the original-cased tokens are
                    # written to the pretranslations.

                    segments = [(lowercase(row.source_segment), lowercase(row.target_segment)) for _, row in pt_batch]
                    if transductive_model is not None:
                        # The pretranslations are the first rows of the training corpus, so their
                        # alignments were already computed during training.

                        alignments = [
                            transductive_model.get_training_alignment(index + i) for i in range(len(pt_batch))
                        ]
                    else:
                        alignments = alignment_model.align_batch(segments)
                    for (pt_info, row), (source_segment, target_segment), alignment in zip(
                        pt_batch, segments, alignments, strict=True
                    ):
                        word_pairs = alignment.to_aligned_word_pairs(include_null=True)
                        alignment_model.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)
                        pt_info["sourceTokens"] = list(row.source_segment)
                        pt_info["translationTokens"] = list(row.target_segment)
                        pt_info["alignment"] = AlignedWordPair.to_string(word_pairs, include_scores=True)
                        yield pt_info
                    index += len(pt_batch)

    def _save_model(self) -> None:
        return


def _read_pretranslations(path: Path) -> Generator[PretranslationInfo, None, None]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
