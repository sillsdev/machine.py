import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional, Tuple

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..utils.phased_progress_reporter import PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .eflomal_aligner import EflomalAligner, is_eflomal_available, tokenize
from .translation_file_service import PretranslationInfo, TranslationFileService

logger = logging.getLogger(__name__)


class TranslationEngineBuildJob(ABC):
    def __init__(self, config: Any, translation_file_service: TranslationFileService) -> None:
        self._config = config
        self._translation_file_service = translation_file_service

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, float]:
        if check_canceled is not None:
            check_canceled()

        source_corpus = self._translation_file_service.create_source_corpus()
        target_corpus = self._translation_file_service.create_target_corpus()
        parallel_corpus: ParallelTextCorpus = source_corpus.align_rows(target_corpus)

        parallel_corpus_size = parallel_corpus.count(include_empty=False)
        progress_reporter = self._get_progress_reporter(progress, parallel_corpus_size)

        if parallel_corpus_size == 0:
            train_corpus_size, confidence = self._respond_to_no_training_corpus()
        else:
            train_corpus_size, confidence = self._train_model(
                source_corpus, target_corpus, parallel_corpus, progress_reporter, check_canceled
            )

        if check_canceled is not None:
            check_canceled()

        logger.info("Pretranslating segments")
        self._batch_inference(progress_reporter, check_canceled)

        if "align_pretranslations" in self._config and self._config.align_pretranslations and is_eflomal_available():
            logger.info("Aligning source to pretranslations")
            self._align(progress_reporter, check_canceled)

        self._save_model()
        return train_corpus_size, confidence

    @abstractmethod
    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]], corpus_size: int
    ) -> PhasedProgressReporter: ...

    @abstractmethod
    def _respond_to_no_training_corpus(
        self,
    ) -> Tuple[int, float]: ...

    @abstractmethod
    def _train_model(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        parallel_corpus: ParallelTextCorpus,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> Tuple[int, float]: ...

    @abstractmethod
    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None: ...

    def _align(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None:
        if check_canceled is not None:
            check_canceled()

        logger.info("Aligning source to pretranslations")
        with ExitStack() as stack:
            # phase_progress = stack.enter_context(progress_reporter.start_next_phase())
            progress_reporter.start_next_phase()

            src_tokenized = [
                tokenize(s["pretranslation"])
                for s in stack.enter_context(self._translation_file_service.get_source_pretranslations())
            ]
            trg_info = [
                pt_info for pt_info in stack.enter_context(self._translation_file_service.get_target_pretranslations())
            ]
            trg_tokenized = [tokenize(pt_info["pretranslation"]) for pt_info in trg_info]

            with TemporaryDirectory() as td:
                aligner = EflomalAligner(Path(td))
                logger.info("Training aligner")
                aligner.train(src_tokenized, trg_tokenized)

                if check_canceled is not None:
                    check_canceled()

                logger.info("Aligning pretranslations")
                alignments = aligner.align()

            if check_canceled is not None:
                check_canceled()

            writer = stack.enter_context(self._translation_file_service.open_target_pretranslation_writer())
            for trg_pi, src_toks, trg_toks, alignment in zip(trg_info, src_tokenized, trg_tokenized, alignments):
                writer.write(
                    PretranslationInfo(
                        corpusId=trg_pi["corpusId"],
                        textId=trg_pi["textId"],
                        refs=trg_pi["refs"],
                        pretranslation=trg_pi["pretranslation"],
                        source_toks=list(src_toks),
                        pretranslation_toks=list(trg_toks),
                        alignment=alignment,
                    )
                )

    @abstractmethod
    def _save_model(self) -> None: ...
