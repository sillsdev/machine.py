import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, overload

import thot.alignment as ta

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.parallel_text_segment import ParallelTextSegment
from ...corpora.token_processors import NO_OP, TokenProcessor
from ...utils.progress_status import ProgressStatus
from ...utils.typeshed import StrPath
from ..trainer import Trainer, TrainStats
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_parameters import ThotWordAlignmentParameters


class ThotWordAlignmentModelTrainer(Trainer):
    @overload
    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        corpus: ParallelTextCorpus,
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
        max_corpus_count: int = sys.maxsize,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        corpus: Tuple[StrPath, StrPath],
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
    ) -> None:
        ...

    def __init__(
        self,
        model_type: ThotWordAlignmentModelType,
        corpus: Union[ParallelTextCorpus, Tuple[StrPath, StrPath]],
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
        max_corpus_count: int = sys.maxsize,
    ) -> None:
        if isinstance(corpus, tuple) and max_corpus_count != sys.maxsize:
            raise ValueError("max_corpus_count cannot be set when corpus filenames are provided.")

        self._prefix_filename = None if prefix_filename is None else Path(prefix_filename)
        self._source_preprocessor = source_preprocessor
        self._target_preprocessor = target_preprocessor
        self._parallel_corpus = corpus
        self._max_corpus_count = max_corpus_count
        self._stats = TrainStats()

        def null_segment_filter(s: ParallelTextSegment, i: int) -> bool:
            return True

        self._segment_filter = null_segment_filter

        self._models: List[Tuple[ta.AlignmentModel, int]] = []
        if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            fast_align = ta.FastAlignModel()
            fast_align.variational_bayes = parameters.get_variational_bayes(model_type)
            if parameters.fast_align_p0 is not None:
                fast_align.fast_align_p0 = parameters.fast_align_p0
            self._models.append((fast_align, parameters.get_fast_align_iteration_count(model_type)))
        else:
            ibm1 = ta.Ibm1AlignmentModel()
            ibm1.variational_bayes = parameters.get_variational_bayes(model_type)
            self._models.append((ibm1, parameters.get_ibm1_iteration_count(model_type)))

            ibm2_or_hmm: Optional[ta.AlignmentModel] = None
            if model_type >= ThotWordAlignmentModelType.IBM2:
                if parameters.get_hmm_iteration_count(model_type) > 0:
                    ibm2_or_hmm = ta.HmmAlignmentModel(ibm1)
                    if parameters.hmm_p0 is not None:
                        ibm2_or_hmm.hmm_p0 = parameters.hmm_p0
                    if parameters.hmm_lexical_smoothing_factor is not None:
                        ibm2_or_hmm.lexical_smoothing_factor = parameters.hmm_lexical_smoothing_factor
                    if parameters.hmm_alignment_smoothing_factor is not None:
                        ibm2_or_hmm.hmm_alignment_smoothing_factor = parameters.hmm_alignment_smoothing_factor
                    self._models.append((ibm2_or_hmm, parameters.get_hmm_iteration_count(model_type)))
                else:
                    ibm2_or_hmm = ta.Ibm2AlignmentModel(ibm1)
                    self._models.append((ibm2_or_hmm, parameters.get_ibm2_iteration_count(model_type)))

            ibm3: Optional[ta.Ibm3AlignmentModel] = None
            if (
                model_type >= ThotWordAlignmentModelType.IBM3
                and ibm2_or_hmm is not None
                and parameters.get_ibm3_iteration_count(model_type) > 0
            ):
                ibm3 = ta.Ibm3AlignmentModel(ibm2_or_hmm)
                if parameters.ibm3_fertility_smoothing_factor is not None:
                    ibm3.fertility_smoothing_factor = parameters.ibm3_fertility_smoothing_factor
                if parameters.ibm3_count_threshold is not None:
                    ibm3.count_threshold = parameters.ibm3_count_threshold
                self._models.append((ibm3, parameters.get_ibm3_iteration_count(model_type)))

            if model_type >= ThotWordAlignmentModelType.IBM4:
                ibm4: Optional[ta.Ibm4AlignmentModel] = None
                if ibm3 is not None:
                    ibm4 = ta.Ibm4AlignmentModel(ibm3)
                elif isinstance(ibm2_or_hmm, ta.HmmAlignmentModel):
                    ibm4 = ta.Ibm4AlignmentModel(ibm2_or_hmm)
                if ibm4 is not None:
                    if parameters.ibm4_distortion_smoothing_factor is not None:
                        ibm4.distortion_smoothing_factor = parameters.ibm4_distortion_smoothing_factor
                    for word, word_class in parameters.source_word_classes.items():
                        ibm4.map_src_word_to_word_class(word, word_class)
                    for word, word_class in parameters.target_word_classes.items():
                        ibm4.map_trg_word_to_word_class(word, word_class)
                    self._models.append((ibm4, parameters.get_ibm4_iteration_count(model_type)))

        self._max_segment_length = self._model.max_sentence_length

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

    @property
    def _model(self) -> ta.AlignmentModel:
        return self._models[-1][0]

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        num_steps = sum(iterations + 1 for _, iterations in self._models if iterations > 0) + 1
        cur_step = 0

        if progress is not None:
            progress(ProgressStatus.from_step(cur_step, num_steps))

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
        cur_step += 1
        if progress is not None:
            progress(ProgressStatus.from_step(cur_step, num_steps))
        if check_canceled is not None:
            check_canceled()

        trained_segment_count = 0
        for model, iteration_count in self._models:
            if iteration_count == 0:
                continue

            trained_segment_count = model.start_training()
            cur_step += 1
            if progress is not None:
                progress(ProgressStatus.from_step(cur_step, num_steps))
            if check_canceled is not None:
                check_canceled()

            for _ in range(iteration_count):
                model.train()
                cur_step += 1
                if progress is not None:
                    progress(ProgressStatus.from_step(cur_step, num_steps))
                if check_canceled is not None:
                    check_canceled()
            model.end_training()
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
