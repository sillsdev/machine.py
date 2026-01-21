import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..utils.phased_progress_reporter import PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .translation_file_service import TranslationFileService

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

    @abstractmethod
    def _save_model(self) -> None: ...
