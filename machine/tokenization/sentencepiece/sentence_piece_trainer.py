from typing import Callable, Optional

import sentencepiece as sp

from ...corpora.text_corpus import TextCorpus
from ...translation.trainer import Trainer, TrainStats
from ...utils.progress_status import ProgressStatus


class SentencePieceTrainer(Trainer):
    def __init__(self, corpus: Optional[TextCorpus] = None, **kwargs) -> None:
        self._corpus = corpus
        self._kwargs = kwargs
        self._stats = TrainStats()

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._corpus is not None:
            with self._corpus.filter_nonempty().get_rows() as rows:
                sp.SentencePieceTrainer.Train(sentence_iterator=(row.text for row in rows), **self._kwargs)
        else:
            sp.SentencePieceTrainer.Train(**self._kwargs)

    def save(self) -> None:
        ...

    def stats(self) -> TrainStats:
        raise NotImplementedError
