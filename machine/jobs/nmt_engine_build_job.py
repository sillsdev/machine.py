import json
import logging
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Iterable, Optional, Sequence, Tuple

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpora_utils import batch
from ..corpora.flatten import flatten
from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..corpora.text_file_text_corpus import TextFileTextCorpus
from ..corpora.token_processors import lowercase
from ..tokenization.tokenizer_factory import create_tokenizer
from ..translation.transductive_word_alignment_model import TransductiveWordAlignmentModel
from ..translation.translation_engine import TranslationEngine
from ..translation.word_alignment_matrix import WordAlignmentMatrix
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
        word_alignment_model_factory: WordAlignmentModelFactory,
    ) -> None:
        self._nmt_model_factory = nmt_model_factory
        self._nmt_model_factory.init()
        self._alignment_model_factory = word_alignment_model_factory
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
            src_pretranslations = stack.enter_context(self._translation_file_service.get_source_pretranslations())
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
            source_segments = [pt_info["translation"] for pt_info in pt_batch]
            for pt_info, result in zip(pt_batch, engine.translate_batch(source_segments), strict=True):
                pt_info["translation"] = result.translation
                pt_info["sequenceConfidence"] = result.sequence_confidence
            current_inference_step += len(pt_batch)
            phase_progress(ProgressStatus.from_step(current_inference_step, inference_step_count))
            yield from zip(source_segments, pt_batch)

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
                        alignments: Sequence[WordAlignmentMatrix] = []
                        for i in range(len(pt_batch)):
                            logger.info(f"Index: {index}, i: {i}, pt_info: {pt_batch[i]}")
                            alignments.append(transductive_model.get_training_alignment(index + i))

                    else:
                        alignments = alignment_model.align_batch(segments)
                    for (pt_info, row), (source_segment, target_segment), alignment in zip(
                        pt_batch, segments, alignments, strict=True
                    ):
                        word_pairs = alignment.to_aligned_word_pairs(include_null=False)
                        alignment_model.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)
                        pt_info["sourceTokens"] = list(row.source_segment)
                        pt_info["translationTokens"] = list(row.target_segment)
                        pt_info["alignment"] = AlignedWordPair.to_string(word_pairs, include_scores=True)
                        yield pt_info
                    index += len(pt_batch)

    def _save_model(self) -> None:
        if "save_model" in self._config and self._config.save_model is not None:
            logger.info("Saving model")
            model_path = self._nmt_model_factory.save_model()
            self._translation_file_service.save_model(
                model_path, f"models/{self._config.save_model + ''.join(model_path.suffixes)}"
            )


def _read_pretranslations(path: Path) -> Generator[PretranslationInfo, None, None]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
