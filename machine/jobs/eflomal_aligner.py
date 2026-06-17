# NOTE: this is a temporary solution to be able to use the eflomal aligner inside of machine.py.
# The vast majority of this code is taken from the silnlp repository.

import os
import subprocess
from contextlib import ExitStack
from importlib.util import find_spec
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Iterable, List, Sequence, Tuple

from ..corpora import AlignedWordPair
from ..corpora.token_processors import escape_spaces, lowercase, normalize
from ..tokenization import LatinWordTokenizer
from ..translation import SymmetrizationHeuristic, WordAlignmentMatrix


# From silnlp.common.package_utils
def is_eflomal_available() -> bool:
    return find_spec("eflomal") is not None


if is_eflomal_available():
    from eflomal import read_text, write_text  # type: ignore

EFLOMAL_PATH = Path(os.getenv("EFLOMAL_PATH", "."), "eflomal")
TOKENIZER = LatinWordTokenizer()


# From silnlp.alignment.tools
def execute_eflomal(
    source_path: Path,
    target_path: Path,
    forward_links_path: Path,
    reverse_links_path: Path,
    forward_scores_output: Path,
    reverse_scores_output: Path,
    n_iterations: Tuple[int, int, int],
) -> None:
    if not is_eflomal_available():
        raise RuntimeError("eflomal is not installed.")

    args = [
        str(EFLOMAL_PATH),
        "-s",
        str(source_path),
        "-t",
        str(target_path),
        "-f",
        str(forward_links_path),
        "-r",
        str(reverse_links_path),
        "-F",
        str(forward_scores_output),
        "-R",
        str(reverse_scores_output),
        # "-q",
        "-m",
        "3",
        "-n",
        "3",
        "-N",
        "0.2",
        "-1",
        str(n_iterations[0]),
        "-2",
        str(n_iterations[1]),
        "-3",
        str(n_iterations[2]),
    ]
    subprocess.run(args, stderr=subprocess.DEVNULL)


# From silnlp.alignment.eflomal
def to_word_alignment_matrix(alignment_str: str) -> WordAlignmentMatrix:
    word_pairs = AlignedWordPair.from_string(alignment_str)
    row_count = 0
    column_count = 0
    for pair in word_pairs:
        if pair.source_index + 1 > row_count:
            row_count = pair.source_index + 1
        if pair.target_index + 1 > column_count:
            column_count = pair.target_index + 1
    return WordAlignmentMatrix.from_word_pairs(row_count, column_count, word_pairs)


# From silnlp.alignment.eflomal
def to_eflomal_text_file(input: Iterable[str], output_file: IO[bytes], prefix_len: int = 0, suffix_len: int = 0) -> int:
    sents, index = read_text(input, True, prefix_len, suffix_len)
    n_sents = len(sents)
    voc_size = len(index)
    write_text(output_file, tuple(sents), voc_size)
    return n_sents


# From silnlp.alignment.eflomal
def prepare_files(
    src_input: Iterable[str], src_output_file: IO[bytes], trg_input: Iterable[str], trg_output_file: IO[bytes]
) -> int:
    n_src_sents = to_eflomal_text_file(src_input, src_output_file)
    n_trg_sents = to_eflomal_text_file(trg_input, trg_output_file)
    if n_src_sents != n_trg_sents:
        raise ValueError("Mismatched file sizes")
    return n_src_sents


def tokenize(sent: str) -> Sequence[str]:
    return list(TOKENIZER.tokenize(sent))


def normalize_for_alignment(sent: Sequence[str]) -> str:
    return " ".join(lowercase(normalize("NFC", escape_spaces(sent))))


def compute_aligned_word_pair_scores(
    forward_matrix: WordAlignmentMatrix,
    forward_sentence_score: float,
    reverse_sentence_score: float,
) -> str:
    # Get the sentence score as 0.0-1.0, from the average logp sentence score
    avg_logp = (forward_sentence_score + reverse_sentence_score) / 2.0
    sentence_score = 1.0 / (1.0 + abs(avg_logp))

    scored: List[AlignedWordPair] = []
    forward_pairs = forward_matrix.to_aligned_word_pairs()
    for word_pair in forward_pairs:
        scored.append(
            AlignedWordPair(
                word_pair.source_index,
                word_pair.target_index,
                translation_score=-1,
                alignment_score=sentence_score,
            )
        )

    return " ".join(str(wp) for wp in scored)


# From silnlp.alignment.eflomal
class EflomalAligner:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir

    def train(self, src_toks: Sequence[Sequence[str]], trg_toks: Sequence[Sequence[str]]) -> None:
        self._model_dir.mkdir(exist_ok=True)
        with TemporaryDirectory() as temp_dir:
            src_eflomal_path = Path(temp_dir, "source")
            trg_eflomal_path = Path(temp_dir, "target")
            with ExitStack() as stack:
                src_output_file = stack.enter_context(src_eflomal_path.open("wb"))
                trg_output_file = stack.enter_context(trg_eflomal_path.open("wb"))
                # Write input files for the eflomal binary
                n_sentences = prepare_files(
                    [normalize_for_alignment(s) for s in src_toks],
                    src_output_file,
                    [normalize_for_alignment(s) for s in trg_toks],
                    trg_output_file,
                )

            iters = max(2, int(round(1.0 * 5000 / sqrt(n_sentences))) if n_sentences > 0 else 0)
            iters4 = max(1, iters // 4)
            n_iterations = (max(2, iters4), iters4, iters)

            # Run wrapper for the eflomal binary
            execute_eflomal(
                src_eflomal_path,
                trg_eflomal_path,
                self._model_dir / "forward-align.txt",
                self._model_dir / "reverse-align.txt",
                self._model_dir / "forward-scores.txt",
                self._model_dir / "reverse-scores.txt",
                n_iterations,
            )

    def align(self, sym_heuristic: str = "grow-diag-final-and") -> List[str]:
        forward_align_path = self._model_dir / "forward-align.txt"
        reverse_align_path = self._model_dir / "reverse-align.txt"
        forward_scores_path = self._model_dir / "forward-scores.txt"
        reverse_scores_path = self._model_dir / "reverse-scores.txt"

        alignments = []
        heuristic = SymmetrizationHeuristic[sym_heuristic.upper().replace("-", "_")]
        with ExitStack() as stack:
            forward_align_file = stack.enter_context(forward_align_path.open("r", encoding="utf-8-sig"))
            reverse_align_file = stack.enter_context(reverse_align_path.open("r", encoding="utf-8-sig"))
            forward_scores_file = stack.enter_context(forward_scores_path.open("r", encoding="utf-8-sig"))
            reverse_scores_file = stack.enter_context(reverse_scores_path.open("r", encoding="utf-8-sig"))

            for forward_align_line, reverse_align_line, forward_sentence_score, reverse_sentence_score in zip(
                forward_align_file, reverse_align_file, forward_scores_file, reverse_scores_file
            ):
                forward_matrix = to_word_alignment_matrix(str(forward_align_line.strip()))
                reverse_matrix = to_word_alignment_matrix(str(reverse_align_line.strip()))
                src_len = max(forward_matrix.row_count, reverse_matrix.row_count)
                trg_len = max(forward_matrix.column_count, reverse_matrix.column_count)

                forward_matrix.resize(src_len, trg_len)
                reverse_matrix.resize(src_len, trg_len)

                forward_matrix.symmetrize_with(reverse_matrix, heuristic)

                scored_word_pairs = compute_aligned_word_pair_scores(
                    forward_matrix,
                    float(forward_sentence_score.strip()),
                    float(reverse_sentence_score.strip()),
                )

                alignments.append(scored_word_pairs)

        return alignments
