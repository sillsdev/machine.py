from __future__ import annotations

import os
import shutil
import sys
from itertools import groupby, repeat
from math import exp, log
from pathlib import Path
from random import Random
from struct import Struct
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple, Union

import numpy as np
import thot.common as tc
import thot.translation as tt

from ...corpora import ParallelTextCorpus, ParallelTextRow
from ...optimization.nelder_mead_simplex import NelderMeadSimplex
from ...statistics.log_space import log_space_add, log_space_divide, to_std_space
from ...utils.progress_status import ProgressStatus
from ...utils.typeshed import StrPath
from .. import MAX_SEGMENT_LENGTH
from ..trainer import Trainer, TrainStats
from .simplex_model_weight_tuner import SimplexModelWeightTuner
from .thot_smt_parameters import ThotSmtParameters, get_thot_smt_parameter
from .thot_train_progress_reporter import ThotTrainProgressReporter
from .thot_utils import escape_token, escape_tokens, load_smt_decoder, load_smt_model
from .thot_word_alignment_model_trainer import ThotWordAlignmentModelTrainer
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_model_utils import create_thot_word_alignment_model
from .thot_word_alignment_parameters import ThotWordAlignmentParameters


def _is_segment_valid(segment: ParallelTextRow) -> bool:
    return (
        not segment.is_empty
        and len(segment.source_segment) <= MAX_SEGMENT_LENGTH
        and len(segment.target_segment) <= MAX_SEGMENT_LENGTH
    )


def _copy_files(src_dir: Path, dest_dir: Path, file_prefix: str) -> None:
    for src_file in src_dir.glob(file_prefix + "*"):
        shutil.copy2(src_file, dest_dir)


def _escape_tokens_row(row: ParallelTextRow) -> ParallelTextRow:
    row.source_segment = list(escape_tokens(row.source_segment))
    row.target_segment = list(escape_tokens(row.target_segment))
    return row


def _prune_lex_table(filename: Path, threshold: float) -> None:
    entries: List[Tuple[int, int, float]] = []
    struct = Struct("IIff")
    with filename.open("rb") as file:
        chunk = file.read(struct.size)
        while chunk != b"":
            entry = struct.unpack(chunk)
            entries.append(entry[:-1])
            chunk = file.read(struct.size)

    if len(entries) == 0:
        return

    with filename.open("wb") as file:
        for _, group in groupby(sorted(entries, key=lambda e: (e[0], -e[2])), key=lambda e: e[0]):
            group_entries = list(group)

            lc_src = group_entries[0][2]
            for entry in group_entries[1:]:
                lc_src = log_space_add(lc_src, entry[2])

            new_lc_src = -99999
            count = 0
            for entry in group_entries:
                prob = to_std_space(log_space_divide(entry[2], lc_src))
                if prob < threshold:
                    break
                new_lc_src = log_space_add(new_lc_src, entry[2])
                count += 1

            for i in range(count):
                entry = group_entries[i]
                file.write(struct.pack(entry[0], entry[1], entry[2], new_lc_src))


def _write_language_model_weights_file(lm_prefix: Path, ngram_size: int, weights: Iterable[float]) -> None:
    weights_str = " ".join(str(round(w, 6)) for w in weights)
    with (lm_prefix.parent / f"{lm_prefix.name}.weights").open("w", encoding="utf-8", newline="\n") as file:
        file.write(f"{ngram_size} 3 10 {weights_str}\n")


def _calculate_perplexity(
    tune_target_corpus: Sequence[Sequence[str]], lm_prefix: Path, ngram_size: int, weights: np.ndarray
) -> float:
    if any(w < 0 or w >= 1.0 for w in weights):
        return 999999

    _write_language_model_weights_file(lm_prefix, ngram_size, weights)
    lp = 0.0
    word_count = 0
    lm = tc.NGramLanguageModel()
    try:
        lm.load(str(lm_prefix))
        for segment in tune_target_corpus:
            lp += lm.get_sentence_log_probability(list(escape_tokens(segment)))
            word_count += len(segment)
    finally:
        lm.clear()
    return exp(-(lp / (word_count + len(tune_target_corpus))) * log(10))


def _filter_phrase_table_using_corpus(filename: Path, source_corpus: Sequence[Sequence[str]]) -> None:
    phrases: Set[str] = set()
    for segment in source_corpus:
        for i in range(len(segment)):
            j = 0
            while j < len(segment) and j + i < len(segment):
                phrase = ""
                for k in range(i, i + j + 1):
                    if k != i:
                        phrase += " "
                    phrase += escape_token(segment[k])
                phrases.add(phrase)
                j += 1

    temp_filename = filename.parent / f"{filename.name}.temp"
    with filename.open("r", encoding="utf-8-sig") as file, temp_filename.open(
        "w", encoding="utf-8", newline="\n"
    ) as temp_file:
        for line in file:
            fields = line.strip().split("|||")
            phrase = fields[1].strip()
            if phrase in phrases:
                temp_file.write(line)
    shutil.copy2(temp_filename, filename)
    os.remove(temp_filename)


class ThotSmtModelTrainer(Trainer):
    def __init__(
        self,
        word_alignment_model_type: ThotWordAlignmentModelType,
        corpus: ParallelTextCorpus,
        config: Optional[Union[ThotSmtParameters, StrPath]] = None,
    ) -> None:
        if config is None:
            config = ThotSmtParameters()
        if isinstance(config, ThotSmtParameters):
            self._config_filename = None
            parameters = config
        else:
            self._config_filename = Path(config)
            parameters = ThotSmtParameters.load(config)
        self._parameters = parameters
        self._word_alignment_model_type = word_alignment_model_type
        self._corpus = corpus
        self._model_weight_tuner = SimplexModelWeightTuner(word_alignment_model_type)

        self._temp_dir = TemporaryDirectory(prefix="thot-smt-train-")

        self._train_lm_dir = Path(self._temp_dir.name, "lm")
        self._train_tm_dir = Path(self._temp_dir.name, "tm_train")
        self._tune_tm_dir = Path(self._temp_dir.name, "tm_tune")

        self._lm_file_prefix = os.path.basename(self._parameters.language_model_filename_prefix)
        self._tm_file_prefix = os.path.basename(self._parameters.translation_model_filename_prefix)
        self._stats = TrainStats()
        self.max_corpus_count = sys.maxsize
        self.seed = 31415

    @property
    def config_filename(self) -> Optional[Path]:
        return self._config_filename

    @property
    def parameters(self) -> ThotSmtParameters:
        return self._parameters

    @property
    def stats(self) -> TrainStats:
        return self._stats

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        reporter = ThotTrainProgressReporter(progress, check_canceled)

        train_corpus, test_corpus, train_count, test_count = (
            self._corpus.filter(_is_segment_valid)
            .take(self.max_corpus_count)
            .split(percent=0.1, size=1000, seed=self.seed)
        )

        self._train_lm_dir.mkdir()
        train_lm_prefix = self._train_lm_dir / self._lm_file_prefix
        self._train_tm_dir.mkdir()
        train_tm_prefix = self._train_tm_dir / self._tm_file_prefix

        with reporter.start_next_phase():
            self._train_language_model(train_lm_prefix, 3, train_corpus)

        self._train_translation_model(train_tm_prefix, train_corpus, train_count, reporter)

        reporter.check_canceled()

        self._tune_tm_dir.mkdir()
        tune_tm_prefix = self._tune_tm_dir / self._tm_file_prefix
        _copy_files(self._train_tm_dir, self._tune_tm_dir, self._tm_file_prefix)

        tune_source_corpus: List[Sequence[str]] = []
        tune_target_corpus: List[Sequence[str]] = []
        with test_corpus.get_rows() as rows:
            for row in rows:
                tune_source_corpus.append(row.source_segment)
                tune_target_corpus.append(row.target_segment)

        with reporter.start_next_phase():
            self._tune_language_model(train_lm_prefix, tune_target_corpus, 3)

        with reporter.start_next_phase() as phase_progress:
            self._tune_translation_model(
                tune_tm_prefix, train_lm_prefix, tune_source_corpus, tune_target_corpus, phase_progress
            )

        with reporter.start_next_phase() as phase_progress:
            self._train_tune_corpus(
                train_tm_prefix, train_lm_prefix, tune_source_corpus, tune_target_corpus, phase_progress
            )

        self.stats.train_corpus_size = train_count + test_count

    def close(self) -> None:
        self._temp_dir.cleanup()

    def __enter__(self) -> ThotSmtModelTrainer:
        return self

    def _train_language_model(self, lm_prefix: Path, ngram_size: int, train_corpus: ParallelTextCorpus) -> None:
        self._write_ngram_counts_file(lm_prefix, ngram_size, train_corpus)
        _write_language_model_weights_file(lm_prefix, ngram_size, repeat(0.5, ngram_size * 3))
        self._write_word_prediction_file(lm_prefix, train_corpus)

    def _write_ngram_counts_file(self, lm_prefix: Path, ngram_size: int, train_corpus: ParallelTextCorpus) -> None:
        word_count = 0
        ngrams: Dict[Tuple[str, ...], int] = {}
        vocab: Set[str] = set()
        with train_corpus.get_rows() as rows:
            for row in rows:
                words = ["<s>"]
                for word in (escape_token(w) for w in row.target_segment):
                    if word in vocab:
                        words.append(word)
                    else:
                        vocab.add(word)
                        words.append("<unk>")
                words.append("</s>")
                if len(words) == 2:
                    continue
                word_count += len(words)
                for n in range(1, ngram_size + 1):
                    for i in range(len(words) - n + 1):
                        ngram = tuple(words[j] for j in range(i, i + n))
                        count = ngrams.get(ngram)
                        ngrams[ngram] = 1 if count is None else count + 1

        with lm_prefix.open("w", encoding="utf-8", newline="\n") as file:
            for ngram, count in sorted(ngrams.items(), key=lambda item: (len(item[0]), " ".join(item[0]))):
                ngram_str = " ".join(ngram)
                prefix_count = word_count if len(ngram) == 1 else ngrams[ngram[:-1]]
                file.write(f"{ngram_str} {prefix_count} {count}\n")

    def _write_word_prediction_file(self, lm_prefix: Path, train_corpus: ParallelTextCorpus) -> None:
        rand = Random(self.seed)
        with (lm_prefix.parent / f"{lm_prefix.name}.wp").open(
            "w", encoding="utf-8", newline="\n"
        ) as file, train_corpus.take(100000).get_rows() as rows:
            for row in sorted(rows, key=lambda _: rand.randint(0, sys.maxsize)):
                segment_str = " ".join(escape_token(t) for t in row.target_segment)
                file.write(segment_str + "\n")

    def _train_translation_model(
        self, tm_prefix: Path, train_corpus: ParallelTextCorpus, train_count: int, reporter: ThotTrainProgressReporter
    ) -> None:
        invswm_prefix = tm_prefix.parent / f"{tm_prefix.name}_invswm"
        self._generate_word_alignment_model(invswm_prefix, train_corpus, train_count, reporter)

        swm_prefix = tm_prefix.parent / f"{tm_prefix.name}_swm"
        self._generate_word_alignment_model(swm_prefix, train_corpus.invert(), train_count, reporter)

        with reporter.start_next_phase():
            extractor = tt.AlignmentExtractor()
            try:
                extractor.open(str(swm_prefix) + ".bestal")
                extractor.symmetrize1(str(invswm_prefix) + ".bestal", str(tm_prefix) + ".A3.final", transpose=True)
            finally:
                extractor.close()

        with reporter.start_next_phase():
            phrase_model = tt.PhraseModel()
            try:
                params = tt.PhraseExtractParameters()
                params.max_target_phrase_length = 10
                phrase_model.build(str(tm_prefix) + ".A3.final", params, pseudo_ml=False)
                phrase_model.print_phrase_table(str(tm_prefix) + ".ttable", n=20)
            finally:
                phrase_model.clear()

        with (tm_prefix.parent / f"{tm_prefix.name}.lambda").open("w", encoding="utf-8") as file:
            file.write("0.7 0.7")
        with (tm_prefix.parent / f"{tm_prefix.name}.srcsegmlentable").open("w", encoding="utf-8") as file:
            file.write("Uniform")
        with (tm_prefix.parent / f"{tm_prefix.name}.trgcutstable").open("w", encoding="utf-8") as file:
            file.write("0.999")
        with (tm_prefix.parent / f"{tm_prefix.name}.trgsegmlentable").open("w", encoding="utf-8") as file:
            file.write("Geometric")

    def _generate_word_alignment_model(
        self, swm_prefix: Path, train_corpus: ParallelTextCorpus, train_count: int, reporter: ThotTrainProgressReporter
    ) -> None:
        with reporter.start_next_phase() as phase_progress:
            self._train_word_alignment_model(swm_prefix, train_corpus, phase_progress, reporter.check_canceled)

        reporter.check_canceled()

        ext: Optional[str] = None
        if self._word_alignment_model_type is ThotWordAlignmentModelType.HMM:
            ext = ".hmm_lexnd"
        elif (
            self._word_alignment_model_type is ThotWordAlignmentModelType.IBM1
            or self._word_alignment_model_type is ThotWordAlignmentModelType.IBM2
        ):
            ext = ".ibm_lexnd"
        elif self._word_alignment_model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            ext = ".fa_lexnd"
        assert ext is not None

        _prune_lex_table(swm_prefix.parent / (swm_prefix.name + ext), 0.00001)

        with reporter.start_next_phase() as phase_progress:
            self._generate_best_alignments(
                swm_prefix, swm_prefix.parent / f"{swm_prefix.name}.bestal", train_corpus, train_count, phase_progress
            )

    def _train_word_alignment_model(
        self,
        swm_prefix: Path,
        train_corpus: ParallelTextCorpus,
        progress: Callable[[ProgressStatus], None],
        check_canceled: Callable[[], None],
    ) -> None:
        parameters = ThotWordAlignmentParameters()
        if self._word_alignment_model_type is ThotWordAlignmentModelType.FAST_ALIGN:
            parameters.fast_align_iteration_count = self._parameters.learning_em_iters
        else:
            parameters.ibm1_iteration_count = self._parameters.learning_em_iters
            parameters.ibm2_iteration_count = (
                self._parameters.learning_em_iters
                if self._word_alignment_model_type is ThotWordAlignmentModelType.IBM2
                else 0
            )
            parameters.hmm_iteration_count = self._parameters.learning_em_iters
            parameters.ibm3_iteration_count = self._parameters.learning_em_iters
            parameters.ibm4_iteration_count = self._parameters.learning_em_iters

        trainer = ThotWordAlignmentModelTrainer(self._word_alignment_model_type, train_corpus, swm_prefix, parameters)
        trainer.train(progress, check_canceled)
        trainer.save()

    def _generate_best_alignments(
        self,
        swm_prefix: Path,
        filename: Path,
        train_corpus: ParallelTextCorpus,
        train_count: int,
        progress: Callable[[ProgressStatus], None],
    ) -> None:
        model = create_thot_word_alignment_model(self._word_alignment_model_type)
        model.load(swm_prefix)
        with filename.open("w", encoding="utf-8", newline="\n") as file, train_corpus.transform(
            _escape_tokens_row
        ).get_rows() as rows:
            i = 0
            for row in rows:
                file.write("# 1\n")
                file.write(model.get_giza_format_string(row))
                i += 1
                progress(ProgressStatus.from_step(i, train_count))

    def _tune_language_model(self, lm_prefix: Path, tune_target_corpus: List[Sequence[str]], ngram_size: int) -> None:
        if len(tune_target_corpus) == 0:
            return

        simplex = NelderMeadSimplex(convergence_tolerance=0.1, max_function_evaluations=200, scale=1.0)
        result = simplex.find_minimum(
            lambda w, _: _calculate_perplexity(tune_target_corpus, lm_prefix, ngram_size, w),
            list(repeat(0.5, ngram_size * 3)),
        )
        _write_language_model_weights_file(lm_prefix, ngram_size, result.minimizing_point)
        self.stats.metrics["perplexity"] = result.error_value

    def _tune_translation_model(
        self,
        tune_tm_prefix: Path,
        tune_lm_prefix: Path,
        tune_source_corpus: List[Sequence[str]],
        tune_target_corpus: List[Sequence[str]],
        progress: Callable[[ProgressStatus], None],
    ) -> None:
        if len(tune_source_corpus) == 0:
            self.stats.metrics["bleu"] = 0.0
            return

        phrase_table_filename = tune_tm_prefix.parent / f"{tune_tm_prefix.name}.ttable"
        _filter_phrase_table_using_corpus(phrase_table_filename, tune_source_corpus)

        old_parameters = self._parameters
        initial_parameters = old_parameters.copy()
        initial_parameters.translation_model_filename_prefix = str(tune_tm_prefix)
        initial_parameters.language_model_filename_prefix = str(tune_lm_prefix)
        initial_parameters.model_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

        tuned_parameters = self._model_weight_tuner.tune(
            initial_parameters, tune_source_corpus, tune_target_corpus, self.stats, progress
        )
        self._parameters = tuned_parameters
        self._parameters.translation_model_filename_prefix = old_parameters.translation_model_filename_prefix
        self._parameters.language_model_filename_prefix = old_parameters.language_model_filename_prefix

    def _train_tune_corpus(
        self,
        train_tm_prefix: Path,
        train_lm_prefix: Path,
        tune_source_corpus: List[Sequence[str]],
        tune_target_corpus: List[Sequence[str]],
        progress: Callable[[ProgressStatus], None],
    ) -> None:
        if len(tune_source_corpus) == 0:
            return

        parameters = self._parameters.copy()
        parameters.translation_model_filename_prefix = str(train_tm_prefix)
        parameters.language_model_filename_prefix = str(train_lm_prefix)
        smt_model: Optional[tt.SmtModel] = None
        decoder: Optional[tt.SmtDecoder] = None
        try:
            smt_model = load_smt_model(self._word_alignment_model_type, parameters)
            decoder = load_smt_decoder(smt_model, parameters)
            for i in range(len(tune_source_corpus)):
                if i > 0:
                    progress(ProgressStatus.from_step(i, len(tune_source_corpus)))
                decoder.train_sentence_pair(" ".join(tune_source_corpus[i]), " ".join(tune_target_corpus[i]))
            progress(ProgressStatus.from_step(len(tune_source_corpus), len(tune_source_corpus)))
            smt_model.print_translation_model(parameters.translation_model_filename_prefix)
            smt_model.print_language_model(parameters.language_model_filename_prefix)
        finally:
            if decoder is not None:
                decoder.clear()
            if smt_model is not None:
                smt_model.clear()

    def save(self) -> None:
        self._save_parameters()

        lm_dir = Path(self._parameters.language_model_filename_prefix).parent
        tm_dir = Path(self._parameters.translation_model_filename_prefix).parent

        lm_dir.mkdir(exist_ok=True)
        _copy_files(self._train_lm_dir, lm_dir, self._lm_file_prefix)
        tm_dir.mkdir(exist_ok=True)
        _copy_files(self._train_tm_dir, tm_dir, self._tm_file_prefix)

    def _save_parameters(self) -> None:
        if self._config_filename is None or len(self._parameters.model_weights) == 0:
            return

        with self._config_filename.open("r", encoding="utf-8-sig") as file:
            lines = file.readlines()

        with self._config_filename.open("w", encoding="utf-8", newline="\n") as file:
            weights_written = False
            for line in lines:
                name, _ = get_thot_smt_parameter(line)
                if name == "tmw":
                    self._write_model_weights(file)
                    weights_written = True
                else:
                    file.write(line)

            if not weights_written:
                self._write_model_weights(file)

    def _write_model_weights(self, file: TextIO) -> None:
        weights_str = " ".join(str(round(w, 6)) for w in self._parameters.model_weights)
        file.write(f"-tmw {weights_str}\n")
