import os
from typing import Callable, Dict, Optional, Sequence, Tuple

from ..corpora.text_corpus import TextCorpus
from ..statistics.conditional_frequency_distribution import ConditionalFrequencyDistribution
from ..tokenization.tokenizer import Tokenizer
from ..tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER
from ..utils.progress_status import ProgressStatus
from ..utils.string_utils import is_delayed_sentence_start, is_sentence_terminal
from .trainer import Trainer, TrainStats
from .truecaser import Truecaser


class UnigramTruecaser(Truecaser):

    def __init__(self, model_path: str = ""):
        self._model_path = model_path
        self._casing: ConditionalFrequencyDistribution = ConditionalFrequencyDistribution()
        self._bestTokens: Dict[str, Tuple[str, int]] = {}

    def __post_init__(self):
        if self._model_path != "":
            self.load(self._model_path)

    def load(self, path: str):
        self._reset()
        self._model_path = path
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                self._parse_line(line.strip())

    def create_trainer(self, corpus: TextCorpus) -> Trainer:
        return UnigramTruecaserTrainer(corpus, self._model_path, self)

    def train_segment(self, segment: Sequence[str], sentence_start: bool = True) -> None:
        for token in segment:
            if is_delayed_sentence_start(token):
                continue

            if not sentence_start and is_sentence_terminal(token):
                sentence_start = True
                continue

            if all(not char.isupper() and not char.islower() for char in token):
                sentence_start = False
                continue

            increment = False
            if not sentence_start:
                increment = True
            elif token[0].islower():
                increment = True

            sentence_start = False

            if increment:
                lower_token = token.lower()
                new_count = self._casing[lower_token].increment(token)
                best_count = 0
                if self._bestTokens.get(lower_token, 0):
                    best_count = self._bestTokens[lower_token][1]
                if new_count > best_count:
                    self._bestTokens[lower_token] = (token, new_count)

    def truecase(self, segment: Sequence[str]) -> Sequence[str]:
        result = []
        for token in segment:
            lower_token = token.lower()
            if self._bestTokens.get(lower_token, 0):
                token = self._bestTokens[lower_token][0]
            result.append(token)
        return result

    def save(self, path: str = "") -> None:
        if path != "":
            self._model_path = path

        with open(self._model_path, "w+", encoding="utf-8") as file:
            for lower_token in self._casing.get_conditions():
                counts = self._casing[lower_token]
                line = " ".join([f"{t} {counts[t]}" for t in counts.get_observed_samples()])
                file.write(f"{line}\n")

    def _reset(self):
        self._casing.reset()
        self._bestTokens.clear()

    def _parse_line(self, line: str):
        parts = line.split()
        for i in range(0, len(parts), 2):
            token = parts[i]
            token_lower = token.lower()
            count = int(parts[i + 1])
            self._casing[parts[i - 1]].increment(token_lower, count)
            best_count = 0
            if self._bestTokens.get(token_lower):
                best_count = self._bestTokens[token_lower][1]
            if count > best_count:
                self._bestTokens[token_lower] = (token, count)


class UnigramTruecaserTrainer(Trainer):

    def __init__(
        self,
        corpus: TextCorpus,
        model_path: str = "",
        new_truecaser: UnigramTruecaser = UnigramTruecaser(),
        tokenizer: Tokenizer = WHITESPACE_TOKENIZER,
    ):
        self.corpus: TextCorpus = corpus
        self.model_path: str = model_path
        self.new_truecaser: UnigramTruecaser = new_truecaser
        self._stats: TrainStats = TrainStats()
        self.tokenizer: Tokenizer = tokenizer

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        step_count = 0
        if progress is not None:
            step_count = self.corpus.count(include_empty=False)
        current_step = 0
        with self.corpus.tokenize(tokenizer=self.tokenizer).filter_nonempty().get_rows() as rows:
            for row in rows:
                if check_canceled is not None:
                    check_canceled()
                self.new_truecaser.train_segment(row)
                current_step += 1
                if progress is not None:
                    progress(ProgressStatus(current_step, step_count))
        self._stats.train_corpus_size = current_step

    def save(self) -> None:
        if self.model_path != "":
            self.new_truecaser.save(self.model_path)

    @property
    def stats(self) -> TrainStats:
        return self._stats
