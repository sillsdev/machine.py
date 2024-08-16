import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..utils.phased_progress_reporter import PhasedProgressReporter
from ..utils.progress_status import ProgressStatus
from .translation_file_service import TranslationFileService

logger = logging.getLogger(__name__)


class TranslationEngineBuildJob(ABC):
    def __init__(self, config: Any, translation_file_service: TranslationFileService) -> None:
        self._config = config
        self._translation_file_service = translation_file_service
        self._train_corpus_size = -1
        self._confidence = -1

    def run(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, float]:
        if check_canceled is not None:
            check_canceled()

        self._start_job()
        self._init_corpus()
        progress_reporter = self._get_progress_reporter(progress)

        if self._parallel_corpus_size == 0:
            self._respond_to_no_training_corpus()
        else:
            self._train_model(progress_reporter, check_canceled)

        if check_canceled is not None:
            check_canceled()

        logger.info("Pretranslating segments")
        self._batch_inference(progress_reporter, check_canceled)

        self._save_model()
        return self._train_corpus_size, self._confidence

    @abstractmethod
    def _start_job(self) -> None: ...

    def _init_corpus(self) -> None:
        logger.info("Downloading data files")
        if "_source_corpus" not in self.__dict__:
            self._source_corpus = self._translation_file_service.create_source_corpus()
            self._target_corpus = self._translation_file_service.create_target_corpus()
            self._parallel_corpus: ParallelTextCorpus = self._source_corpus.align_rows(self._target_corpus)
            self._parallel_corpus_size = self._parallel_corpus.count(include_empty=False)

    @abstractmethod
    def _get_progress_reporter(
        self, progress: Optional[Callable[[ProgressStatus], None]]
    ) -> PhasedProgressReporter: ...

    @abstractmethod
    def _respond_to_no_training_corpus(self) -> None: ...

    @abstractmethod
    def _train_model(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None: ...

    @abstractmethod
    def _batch_inference(
        self,
        progress_reporter: PhasedProgressReporter,
        check_canceled: Optional[Callable[[], None]],
    ) -> None: ...

    @abstractmethod
    def _save_model(self) -> None: ...
