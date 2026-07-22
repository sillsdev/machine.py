import logging
from contextlib import ExitStack
from itertools import chain
from typing import Any, Callable, Optional, Sequence, Tuple

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpora_utils import batch
from ..corpora.dictionary_text_corpus import DictionaryTextCorpus
from ..corpora.memory_text import MemoryText
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_row import TextRow
from ..tokenization.tokenizer_factory import create_tokenizer
from ..utils.phased_progress_reporter import Phase, PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .nmt_model_factory import NmtModelFactory
from .translation_engine_build_job import TranslationEngineBuildJob
from .translation_file_service import PretranslationInfo, TranslationFileService
from .word_alignment_model_factory import WordAlignmentModelFactory

logger = logging.getLogger(__name__)


class NmtEngineBuildJob(TranslationEngineBuildJob):
    def __init__(
        self,
        config: Any,
        nmt_model_factory: NmtModelFactory,
        translation_file_service: TranslationFileService,
        alignment_model_factory: WordAlignmentModelFactory,
    ) -> None:
        self._nmt_model_factory = nmt_model_factory
        self._nmt_model_factory.init()
        self._alignment_model_factory = alignment_model_factory
        self._alignment_model_factory.init()
        super().__init__(config, translation_file_service)

    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter:
        if corpus_size > 0:
            if self._config.align_pretranslations:
                phases = [
                    Phase(message="Training NMT model", percentage=0.8, stage="train"),
                    Phase(message="Pretranslating segments", percentage=0.1, stage="inference"),
                    Phase(message="Aligning segments", percentage=0.1, report_steps=False),
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
                    Phase(message="Aligning segments", percentage=0.1, report_steps=False),
                ]
            else:
                phases = [Phase(message="Pretranslating segments", percentage=1.0, stage="inference")]
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

    def _batch_inference(
        self,
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        logger.info("Pretranslating segments")
        with self._translation_file_service.get_source_pretranslations() as src_pretranslations:
            inference_step_count = sum(1 for _ in src_pretranslations)
        with ExitStack() as stack:
            phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            engine = stack.enter_context(self._nmt_model_factory.create_engine())
            pretranslations = [
                pt_info for pt_info in stack.enter_context(self._translation_file_service.get_source_pretranslations())
            ]
            src_segments = [pt_info["translation"] for pt_info in pretranslations]
            current_inference_step = 0
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            batch_size = self._config["inference_batch_size"]
            for seg_batch in batch(iter(src_segments), batch_size):
                if check_canceled is not None:
                    check_canceled()
                for i, result in enumerate(engine.translate_batch(seg_batch)):
                    pretranslations[current_inference_step + i]["translation"] = result.translation
                    pretranslations[current_inference_step + i]["sequenceConfidence"] = result.sequence_confidence
                current_inference_step += len(seg_batch)
                phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))

            if self._config.align_pretranslations:
                logger.info("Aligning source to pretranslations")
                pretranslations = self._align(
                    src_segments, pretranslations, parallel_training_corpus, progress_reporter, check_canceled
                )

            writer = stack.enter_context(self._translation_file_service.open_target_pretranslation_writer())
            for pretranslation in pretranslations:
                writer.write(pretranslation)

    def _align(
        self,
        src_pretranslate_segments: Sequence[str],
        pretranslations: Sequence[PretranslationInfo],
        parallel_training_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Sequence[PretranslationInfo]:
        if check_canceled is not None:
            check_canceled()

        logger.info("Aligning source to pretranslations")

        tokenizer = create_tokenizer(self._config.thot_align.tokenizer)

        source_inference_corpus = DictionaryTextCorpus(
            MemoryText(
                "pretranslations",
                [
                    TextRow("pretranslations", i, list(tokenizer.tokenize(segment)))
                    for i, segment in enumerate(src_pretranslate_segments)
                ],
            )
        )
        target_inference_corpus = DictionaryTextCorpus(
            MemoryText(
                "pretranslations",
                [
                    TextRow("pretranslations", i, list(tokenizer.tokenize(pretranslation["translation"])))
                    for i, pretranslation in enumerate(pretranslations)
                ],
            )
        )

        parallel_pretranslation_corpus = source_inference_corpus.align_rows(target_inference_corpus)

        alignment_parallel_corpus = ParallelTextCorpus.from_parallel_rows(
            chain(
                parallel_pretranslation_corpus.get_rows(),
                parallel_training_corpus.get_rows(),
            )
        )  # TODO .lowercase()?

        logger.info("Training aligner")
        with (
            progress_reporter.start_next_phase() as phase_progress,
            self._alignment_model_factory.create_model_trainer(tokenizer, alignment_parallel_corpus) as trainer,
        ):
            trainer.train(progress=phase_progress, check_canceled=check_canceled)
            trainer.save()

        logger.info("Aligning pretranslations")
        alignment_model = self._alignment_model_factory.create_alignment_model()

        parallel_pretranslation_rows = list(parallel_pretranslation_corpus)  # TODO .lowercase()?
        alignments = alignment_model.align_batch(parallel_pretranslation_rows)

        all_word_pairs = []
        for parallel_text_row, alignment in zip(parallel_pretranslation_rows, alignments, strict=True):
            word_pairs = alignment.to_aligned_word_pairs(include_null=True)
            alignment_model.compute_aligned_word_pair_scores(
                parallel_text_row.source_segment, parallel_text_row.target_segment, word_pairs
            )
            all_word_pairs.append(word_pairs)

        if check_canceled is not None:
            check_canceled()

        for i in range(len(pretranslations)):
            pretranslations[i]["sourceTokens"] = list(parallel_pretranslation_rows[i].source_segment)
            pretranslations[i]["translationTokens"] = list(parallel_pretranslation_rows[i].target_segment)
            pretranslations[i]["alignment"] = AlignedWordPair.to_string(all_word_pairs[i], include_scores=True)

        return pretranslations

    def _save_model(self) -> None:
        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._nmt_model_factory.save_model()
            self._translation_file_service.save_model(
                model_path, f"models/{self._config.save_model + ''.join(model_path.suffixes)}"
            )
