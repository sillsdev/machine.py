from dataclasses import dataclass, field
from typing import Callable, Optional

from machine.tokenization.tokenizer import Tokenizer
from machine.tokenization.whitespace_tokenizer import WHITESPACE_TOKENIZER

from ..corpora.text_corpus import TextCorpus
from ..utils.progress_status import ProgressStatus
from .trainer import Trainer, TrainStats
from .unigram_truecaser import UnigramTruecaser


@dataclass
class UnigramTruecaserTrainer(Trainer):
    corpus: TextCorpus
    model_path: str = ""
    new_truecaser: UnigramTruecaser = field(default_factory=UnigramTruecaser)
    stats: TrainStats = TrainStats()
    tokenizer: Tokenizer = WHITESPACE_TOKENIZER

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        step_count = 0
        if progress is not None:
            step_count = self.corpus.count(include_empty=False)
        current_step = 0
        for row in self.corpus.tokenize(tokenizer=self.tokenizer).filter_nonempty():
            if check_canceled is not None:
                check_canceled()
            self.new_truecaser.train_segment(row)
            current_step += 1
            if progress is not None:
                progress(ProgressStatus(current_step, step_count))
        self.stats.train_corpus_size = current_step

    def save(self) -> None:
        if self.model_path != "":
            self.new_truecaser.save(self.model_path)


class SubclassUnigramTruecaserTrainer(UnigramTruecaserTrainer):
    _true_caser: UnigramTruecaser

    def save(self):
        self._true_caser._casing = self.new_truecaser._casing
        self._true_caser._bestTokens = self.new_truecaser._bestTokens
        self._true_caser.save()