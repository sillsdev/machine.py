import os
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from machine.corpora.text_corpus import TextCorpus
from machine.statistics.conditional_frequency_distribution import ConditionalFrequencyDistribution
from machine.translation.trainer import Trainer
from machine.translation.truecaser import Truecaser
from machine.utils.string_utils import is_delayed_sentence_start, is_sentence_terminal


@dataclass
class UnigramTruecaser(Truecaser):
    _casing: ConditionalFrequencyDistribution = field(default_factory=ConditionalFrequencyDistribution)
    _bestTokens: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    _model_path: str = ""

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
        from machine.translation.unigram_truecaser_trainer import SubclassUnigramTruecaserTrainer

        return SubclassUnigramTruecaserTrainer(corpus, self._model_path, self)

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
        self._casing = ConditionalFrequencyDistribution()
        self._bestTokens = {}

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
