import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, overload

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.parallel_text_segment import ParallelTextSegment
from ...corpora.token_processors import TokenProcessor
from ...utils.progress_status import ProgressStatus
from ...utils.typeshed import StrPath
from ..trainer import Trainer, TrainStats
from .thot_word_alignment_model_type import ThotWordAlignmentModelType, create_alignment_model


class ThotWordAlignmentModelTrainer(Trainer):
    @overload
    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        prefix_filename: Optional[StrPath],
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        corpus: ParallelTextCorpus,
        max_corpus_count: int = sys.maxsize,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        prefix_filename: Optional[StrPath],
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        corpus: Tuple[StrPath, StrPath],
    ) -> None:
        ...

    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        prefix_filename: Optional[StrPath],
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        corpus: Union[ParallelTextCorpus, Tuple[StrPath, StrPath]],
        max_corpus_count: int = sys.maxsize,
    ) -> None:
        if isinstance(corpus, tuple) and max_corpus_count != sys.maxsize:
            raise ValueError("max_corpus_count cannot be set when corpus filenames are provided.")

        self._model = create_alignment_model(model_type)
        self._prefix_filename = None if prefix_filename is None else Path(prefix_filename)
        self._source_preprocessor = source_preprocessor
        self._target_preprocessor = target_preprocessor
        self._parallel_corpus = corpus
        self._max_corpus_count = max_corpus_count
        self.training_iteration_count = 5
        self._stats = TrainStats()
        self._max_segment_length = self._model.max_sentence_length

        def null_segment_filter(s: ParallelTextSegment, i: int) -> bool:
            return True

        self._segment_filter = null_segment_filter
        if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            self._training_iteration_count = 4

    @property
    def variational_bayes(self) -> bool:
        return self._model.variational_bayes

    @variational_bayes.setter
    def variational_bayes(self, value: bool) -> None:
        self._model.variational_bayes = value

    @property
    def stats(self) -> TrainStats:
        return self._stats

    @property
    def segment_filter(self) -> Callable[[ParallelTextSegment, int], bool]:
        return self._segment_filter

    @segment_filter.setter
    def segment_filter(self, value: Callable[[ParallelTextSegment, int], bool]) -> None:
        if isinstance(self._parallel_corpus, tuple):
            raise RuntimeError("A segment filter cannot be set when corpus filenames are provided.")
        self._segment_filter = value

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        self._model.clear()
        if progress is not None:
            progress(ProgressStatus.from_step(0, self.training_iteration_count + 1))

        if isinstance(self._parallel_corpus, ParallelTextCorpus):
            corpus_count = 0
            index = 0
            for segment in self._parallel_corpus.segments:
                if self._segment_filter(segment, index):
                    source_segment = self._source_preprocessor.process(segment.source_segment)
                    target_segment = self._target_preprocessor.process(segment.target_segment)
                    self._model.add_sentence_pair(source_segment, target_segment, 1)
                    if self._is_segment_valid(segment):
                        corpus_count += 1
                index += 1
                if corpus_count == self._max_corpus_count:
                    break
        else:
            self._model.read_sentence_pairs(str(self._parallel_corpus[0]), str(self._parallel_corpus[1]))

        if check_canceled is not None:
            check_canceled()
        trained_segment_count = self._model.start_training()
        for i in range(self.training_iteration_count):
            if progress is not None:
                progress(ProgressStatus.from_step(i + 1, self.training_iteration_count + 1))
            if check_canceled is not None:
                check_canceled()
            self._model.train()
        self._model.end_training()
        if progress is not None:
            progress(ProgressStatus(1.0))
        self._stats.trained_segment_count = trained_segment_count

    def save(self) -> None:
        if self._prefix_filename is not None:
            self._model.print(str(self._prefix_filename))

    def _is_segment_valid(self, segment: ParallelTextSegment) -> bool:
        return (
            not segment.is_empty
            and len(segment.source_segment) <= self._max_segment_length
            and len(segment.target_segment) <= self._max_segment_length
        )
