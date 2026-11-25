import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Iterable, Optional, Sequence, Tuple, cast

from ..corpora import flatten
from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpora_utils import batch
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..corpora.token_processors import lowercase
from ..tokenization.tokenizer_factory import create_tokenizer
from ..translation.transductive_word_alignment_model import TransductiveWordAlignmentModel
from ..translation.translation_engine import TranslationEngine
from ..translation.word_alignment_matrix import WordAlignmentMatrix
from ..utils.phased_progress_reporter import PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_model_factory import NmtModelFactory
from .translation_engine_build_job import TranslationEngineBuildJob
from .translation_file_service import PretranslationInfo, TranslationFileService, _PostEditedPretranslationInfo
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class NmtEngineBuildJobBase(TranslationEngineBuildJob, ABC):

    def __init__(
        self,
        config: Any,
        nmt_model_factory: NmtModelFactory,
        translation_file_service: TranslationFileService,
        word_alignment_model_factory: WordAlignmentModelFactory,
    ) -> None:
        super().__init__(config, translation_file_service)
        self._nmt_model_factory = nmt_model_factory
        self._alignment_model_factory = word_alignment_model_factory
        self._alignment_model_factory.init()

    def _batch_inference(
        self,
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        with self._translation_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)
        with ExitStack() as stack:
            engine = stack.enter_context(self._nmt_model_factory.create_engine())
            src_pretranslations = stack.enter_context(self._translation_file_service.get_source_pretranslations())
            post_edit_model = self._config.get("post_edit_model")
            if post_edit_model:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is required for postediting.")
                temp_dir = stack.enter_context(TemporaryDirectory())
                drafts_path = Path(temp_dir) / "drafts.jsonl"
                with (
                    progress_reporter.start_next_phase() as phase_progress,
                    drafts_path.open("w", encoding="utf-8") as drafts_file,
                ):
                    for source, pt_info in self._translate(
                        engine, src_pretranslations, inference_step_count, phase_progress, check_canceled
                    ):
                        drafts_file.write(json.dumps([source, pt_info], ensure_ascii=False) + "\n")
                pretranslations = stack.enter_context(
                    closing(
                        self._post_edit(
                            drafts_path,
                            parallel_training_corpus,
                            post_edit_model,
                            api_key,
                            inference_step_count,
                            progress_reporter,
                            check_canceled,
                        )
                    )
                )
            else:
                phase_progress = stack.enter_context(progress_reporter.start_next_phase())
                pretranslations = self._translate(
                    engine, src_pretranslations, inference_step_count, phase_progress, check_canceled
                )
            if self._config.align_pretranslations:
                results: Iterable[PretranslationInfo] = self._align(
                    pretranslations, parallel_training_corpus, progress_reporter, check_canceled
                )
            else:
                results = (pt_info for _, pt_info in pretranslations)

            writer = stack.enter_context(self._translation_file_service.open_target_pretranslation_writer())
            for pt_info in results:
                writer.write(pt_info)

    def _post_edit(
        self,
        drafts_path: Path,
        parallel_training_corpus: ParallelTextCorpus,
        model: str,
        api_key: str,
        count: int,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Generator[Tuple[str, PretranslationInfo], None, None]:
        from ..translation.llm.example import Example
        from ..translation.llm.example_retriever_factory import ExampleRetrieverFactory
        from ..translation.llm.example_retriever_method import ExampleRetrievalMethod
        from ..translation.llm.llm_post_editor import LlmPostEditor
        from ..translation.llm.openrouter_llm_completion_client import OpenRouterLlmCompletionClient
        from ..translation.llm.post_edit_segment import PostEditSegment

        examples = []
        with parallel_training_corpus.get_rows() as rows:
            for row in rows:
                if not row.is_empty:
                    examples.append(Example(row.source_text, row.target_text))
        retriever = ExampleRetrieverFactory.create_example_retriever(ExampleRetrievalMethod.TFIDF, examples)
        with (
            OpenRouterLlmCompletionClient(model=model, api_key=api_key) as client,
            progress_reporter.start_next_phase() as phase_progress,
            drafts_path.open("r", encoding="utf-8") as drafts_file,
        ):
            editor = LlmPostEditor(client, retriever, self._config.src_lang, self._config.trg_lang)
            completed = 0
            for draft_batch in batch((json.loads(line) for line in drafts_file), 64):
                if check_canceled is not None:
                    check_canceled()
                segments = [PostEditSegment(source, info["translation"]) for source, info in draft_batch]

                def report_progress(status: ProgressStatus) -> None:
                    if check_canceled is not None:
                        check_canceled()
                    phase_progress(ProgressStatus.from_step(completed + status.step, count))

                edited = editor.post_edit_batch(segments, max_workers=4, consecutive=False, progress=report_progress)
                for (source, info), result in zip(draft_batch, edited, strict=True):
                    if not result.succeeded and source.strip():
                        logger.warning("Postediting failed for %s: %s", info["textId"], ", ".join(info["sourceRefs"]))
                    edited_info = cast(
                        _PostEditedPretranslationInfo,
                        {**info, "postEditedTranslation": result.translation},
                    )
                    yield source, edited_info
                completed += len(draft_batch)
            logger.info(
                "Postediting usage: %d prompt tokens, %d completion tokens, $%.6f",
                client.total_input_token_count,
                client.total_output_token_count,
                client.total_cost,
            )

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
                    target_translation = pt_info.get("postEditedTranslation")
                    if target_translation is None:
                        target_translation = pt_info["translation"]
                    translation_file.write(target_translation + "\n")
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
            batch_size: int = self._config["inference_batch_size"]
            with (
                self._alignment_model_factory.create_alignment_model() as alignment_model,
                parallel_pretranslation_corpus.tokenize(tokenizer).get_rows() as rows,
            ):
                transductive_model: Optional[TransductiveWordAlignmentModel] = None
                if isinstance(alignment_model, TransductiveWordAlignmentModel):
                    transductive_model = alignment_model
                index = 0
                for pt_batch in batch(zip(_read_pretranslations(pretranslations_path), rows, strict=True), batch_size):
                    if check_canceled is not None:
                        check_canceled()
                    # The aligner is trained on lowercased tokens, so it must also be given
                    # lowercased tokens when aligning and scoring; the original-cased tokens are
                    # written to the pretranslations.
                    segments = [(lowercase(row.source_segment), lowercase(row.target_segment)) for _, row in pt_batch]
                    if transductive_model is not None:
                        # The pretranslations are the first rows of the training corpus, so their
                        # alignments were already computed during training.
                        alignments: Sequence[WordAlignmentMatrix] = [
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

    @abstractmethod
    def _translate(
        self,
        engine: TranslationEngine,
        src_pretranslations: Iterable[PretranslationInfo],
        inference_step_count: int,
        phase_progress: Callable[[ProgressStatus], None],
        check_canceled: Optional[Callable[[], None]],
    ) -> Generator[Tuple[str, PretranslationInfo], None, None]: ...


def join_surrogate_pairs_if_needed(text: str) -> str:
    # Check if any characters are in the UTF-16 surrogate range, and join pairs and drop unpaired halves if needed.
    # These are not supported unjoined by UTF-8 encoding or the MBartTokenizer's Rust implementation.
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    return text


def _read_pretranslations(path: Path) -> Generator[PretranslationInfo, None, None]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
