from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, cast, overload

import thot.alignment as ta

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.parallel_text_row import ParallelTextRow
from ...tokenization.tokenizer import Tokenizer
from ...tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER
from ...utils.progress_status import ProgressStatus
from ...utils.typeshed import StrPath
from ..trainer import Trainer, TrainStats
from .thot_utils import escape_token, escape_tokens
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_parameters import ThotWordAlignmentParameters


class ThotWordAlignmentModelTrainer(Trainer):
    @overload
    def __init__(
        self,
        model_type: Union[ThotWordAlignmentModelType, str],
        corpus: ParallelTextCorpus,
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        max_corpus_count: int = sys.maxsize,
        emit_training_alignments: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model_type: Union[ThotWordAlignmentModelType, str],
        corpus: Tuple[StrPath, StrPath],
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        max_corpus_count: int = sys.maxsize,
        emit_training_alignments: bool = False,
    ) -> None: ...

    def __init__(
        self,
        model_type: Union[ThotWordAlignmentModelType, str],
        corpus: Union[ParallelTextCorpus, Tuple[StrPath, StrPath]],
        prefix_filename: Optional[StrPath],
        parameters: ThotWordAlignmentParameters = ThotWordAlignmentParameters(),
        source_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        target_tokenizer: Tokenizer[str, int, str] = WHITESPACE_TOKENIZER,
        max_corpus_count: int = sys.maxsize,
        emit_training_alignments: bool = False,
    ) -> None:
        if isinstance(corpus, tuple) and max_corpus_count != sys.maxsize:
            raise ValueError("max_corpus_count cannot be set when corpus filenames are provided.")

        self._prefix_filename = None if prefix_filename is None else Path(prefix_filename)
        self._parallel_corpus = corpus
        self._max_corpus_count = max_corpus_count
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.emit_training_alignments = emit_training_alignments
        self._stats = TrainStats()

        if isinstance(model_type, str):
            model_type = ThotWordAlignmentModelType[model_type.upper()]
        self._models: List[Tuple[ta.AlignmentModel, int]] = []
        self._is_eflomal = False
        if model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            fast_align = ta.FastAlignModel()
            fast_align.variational_bayes = parameters.get_variational_bayes(model_type)
            if parameters.fast_align_p0 is not None:
                fast_align.fast_align_p0 = parameters.fast_align_p0
            self._models.append((fast_align, parameters.get_fast_align_iteration_count(model_type)))
        elif model_type is ThotWordAlignmentModelType.EFLOMAL:
            # Eflomal is a single model that runs its own Bayesian IBM1->HMM->fertility cascade.
            self._is_eflomal = True
            eflomal = ta.EflomalAlignmentModel()
            if parameters.eflomal_seed is not None:
                eflomal.seed = parameters.eflomal_seed
            if parameters.eflomal_num_samplers is not None:
                eflomal.num_samplers = parameters.eflomal_num_samplers
            if parameters.eflomal_deterministic is not None:
                eflomal.deterministic = parameters.eflomal_deterministic
            if parameters.eflomal_lex_norm is not None:
                eflomal.lex_norm = parameters.eflomal_lex_norm
            if parameters.is_eflomal_schedule_specified:
                # An explicit schedule turns off the model's automatic (corpus-scaled) schedule.
                eflomal.set_iterations(
                    parameters.get_eflomal_ibm1_iteration_count(),
                    parameters.get_eflomal_hmm_iteration_count(),
                    parameters.get_eflomal_fertility_iteration_count(),
                )
            if parameters.eflomal_lex_alpha is not None:
                eflomal.alpha_lex = parameters.eflomal_lex_alpha
            if parameters.eflomal_jump_alpha is not None:
                eflomal.alpha_jump = parameters.eflomal_jump_alpha
            if parameters.eflomal_fertility_alpha is not None:
                eflomal.alpha_fertility = parameters.eflomal_fertility_alpha
            if parameters.eflomal_p0 is not None:
                eflomal.p0 = parameters.eflomal_p0
            if parameters.eflomal_jump_window is not None:
                eflomal.jump_window = parameters.eflomal_jump_window
            # With an explicit schedule the total sweep count is known up front; with the
            # automatic (corpus-scaled) schedule it is 0 here and resolved from the model after
            # start_training (see train).
            eflomal_iteration_count = (
                parameters.get_eflomal_ibm1_iteration_count()
                + parameters.get_eflomal_hmm_iteration_count()
                + parameters.get_eflomal_fertility_iteration_count()
                if parameters.is_eflomal_schedule_specified
                else 0
            )
            self._models.append((eflomal, eflomal_iteration_count))
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
                        ibm4.map_src_word_to_word_class(escape_token(word), word_class)
                    for word, word_class in parameters.target_word_classes.items():
                        ibm4.map_trg_word_to_word_class(escape_token(word), word_class)
                    self._models.append((ibm4, parameters.get_ibm4_iteration_count(model_type)))

        self._max_segment_length = self._model.max_sentence_length

    @property
    def stats(self) -> TrainStats:
        return self._stats

    @property
    def _model(self) -> ta.AlignmentModel:
        return self._models[-1][0]

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        # One step to load the corpus, then for each trained model one step to start training plus
        # one per training iteration. When the Eflomal model uses its automatic schedule, the
        # iteration count is derived from the corpus during start_training (stored as 0 until then),
        # so the total step count is not known up front; progress is reported as indeterminate until
        # it is resolved below.
        iteration_count_known = not self._is_eflomal or self._models[0][1] > 0
        num_steps: Optional[int] = (
            sum(iterations + 1 for _, iterations in self._models if iterations > 0) + 1
            if iteration_count_known
            else None
        )
        cur_step = 0

        def report() -> None:
            if progress is not None:
                progress(
                    ProgressStatus.from_step(cur_step, num_steps) if num_steps is not None else ProgressStatus(cur_step)
                )

        report()

        if isinstance(self._parallel_corpus, ParallelTextCorpus):
            corpus_count = 0
            index = 0
            with self._parallel_corpus.tokenize(self.source_tokenizer, self.target_tokenizer).get_rows() as rows:
                for row in rows:
                    self._model.add_sentence_pair(
                        list(escape_tokens(row.source_segment)), list(escape_tokens(row.target_segment)), 1
                    )
                    if self._is_segment_valid(row):
                        corpus_count += 1
                    index += 1
                    if corpus_count == self._max_corpus_count:
                        break
        else:
            self._model.read_sentence_pairs(str(self._parallel_corpus[0]), str(self._parallel_corpus[1]))
        cur_step += 1
        report()
        if check_canceled is not None:
            check_canceled()

        if self.emit_training_alignments:
            # Retain the alignments computed during training so that they can be returned without a
            # separate inference pass. Only the final (most refined) model's alignments are needed,
            # since that is the model used for inference.
            self._model.emit_training_alignments = True

        trained_segment_count = 0
        for model, iteration_count in self._models:
            if iteration_count == 0 and not self._is_eflomal:
                continue

            trained_segment_count = model.start_training()

            if self._is_eflomal and iteration_count == 0:
                # The automatic schedule is resolved during start_training; ask the model how many
                # sweeps to run and finalize the (until now indeterminate) total step count.
                iteration_count = cast(ta.EflomalAlignmentModel, model).scheduled_iterations
                num_steps = cur_step + iteration_count + 1

            cur_step += 1
            report()
            if check_canceled is not None:
                check_canceled()

            for _ in range(iteration_count):
                model.train()
                cur_step += 1
                report()
                if check_canceled is not None:
                    check_canceled()
            model.end_training()
        self._stats.train_corpus_size = trained_segment_count

    def save(self) -> None:
        if self._prefix_filename is not None:
            self._model.print(str(self._prefix_filename))

    def close(self) -> None:
        for model, _ in self._models:
            model.clear()
        self._models.clear()

    def __enter__(self) -> ThotWordAlignmentModelTrainer:
        return self

    def _is_segment_valid(self, row: ParallelTextRow) -> bool:
        return (
            not row.is_empty
            and len(row.source_segment) <= self._max_segment_length
            and len(row.target_segment) <= self._max_segment_length
        )
