from abc import abstractmethod
from pathlib import Path
from typing import Collection, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import thot.alignment as ta

from ...corpora import ParallelTextCorpus, TokenProcessor
from ...utils.typeshed import StrPath
from ..ibm1_word_alignment_model import Ibm1WordAlignmentModel
from ..trainer import Trainer
from ..word_alignment_matrix import WordAlignmentMatrix
from .thot_word_alignment_model_trainer import ThotWordAlignmentModelTrainer
from .thot_word_alignment_model_type import ThotWordAlignmentModelType, create_alignment_model

_SPECIAL_SYMBOL_INDICES = {0, 1, 2}


class ThotWordAlignmentModel(Ibm1WordAlignmentModel):
    def __init__(self, prefix_filename: Optional[StrPath] = None, create_new: bool = False) -> None:
        self._set_model(self._create_model())
        if prefix_filename is not None:
            prefix_filename = Path(prefix_filename)
            if create_new or not (prefix_filename.parent / (prefix_filename.name + ".src")).is_file():
                self.create_new(prefix_filename)
            else:
                self.load(prefix_filename)
        else:
            self._prefix_filename = None
        self.training_iteration_count = 5

    @property
    def source_words(self) -> Sequence[str]:
        return self._source_words

    @property
    def target_words(self) -> Sequence[str]:
        return self._target_words

    @property
    def special_symbol_indices(self) -> Collection[int]:
        return _SPECIAL_SYMBOL_INDICES

    @property
    def variational_bayes(self) -> bool:
        return self._model.variational_bayes

    @variational_bayes.setter
    def variational_bayes(self, value: bool) -> None:
        self._model.variational_bayes = value

    @property
    @abstractmethod
    def type(self) -> ThotWordAlignmentModelType:
        ...

    def load(self, prefix_filename: StrPath) -> None:
        prefix_filename = Path(prefix_filename)
        if not (prefix_filename.parent / (prefix_filename.name + ".src")).is_file():
            raise FileNotFoundError("The word alignment model configuration could not be found.")
        self._prefix_filename = prefix_filename
        self._model.clear()
        self._model.load(str(prefix_filename))

    def create_new(self, prefix_filename: StrPath) -> None:
        self._prefix_filename = Path(prefix_filename)
        self._model.clear()

    def save(self) -> None:
        if self._prefix_filename is not None:
            self._model.print(str(self._prefix_filename))

    def create_trainer(
        self,
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        corpus: ParallelTextCorpus,
        max_corpus_count: int,
    ) -> Trainer:
        trainer = _Trainer(
            self, self._prefix_filename, source_preprocessor, target_preprocessor, corpus, max_corpus_count
        )
        trainer.training_iteration_count = self.training_iteration_count
        trainer.variational_bayes = self.variational_bayes
        return trainer

    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        _, best_alignment = self._model.get_best_alignment(source_segment, target_segment)
        return _to_word_alignment_matrix(best_alignment, len(source_segment))

    def get_best_alignments(
        self, source_segments: Sequence[Sequence[str]], target_segments: Sequence[Sequence[str]]
    ) -> Sequence[WordAlignmentMatrix]:
        if len(source_segments) != len(target_segments):
            raise ValueError("The number of source and target segments must be equal.")
        results = self._model.get_best_alignments(source_segments, target_segments)
        matrices: List[WordAlignmentMatrix] = []
        for i in range(len(results)):
            _, best_alignment = results[i]
            src_segment = source_segments[i]
            matrices.append(_to_word_alignment_matrix(best_alignment, len(src_segment)))
        return matrices

    def get_translation_score(
        self, source_word: Optional[Union[str, int]], target_word: Optional[Union[str, int]]
    ) -> float:
        return self.get_translation_probability(source_word, target_word)

    def get_translation_probability(
        self, source_word: Optional[Union[str, int]], target_word: Optional[Union[str, int]]
    ) -> float:
        if source_word is None:
            source_word = 0
        elif isinstance(source_word, str):
            source_word = self._model.get_src_word_index(source_word)
        if target_word is None:
            target_word = 0
        elif isinstance(target_word, str):
            target_word = self._model.get_trg_word_index(target_word)
        return self._model.get_translation_prob(source_word, target_word)

    def get_translations(
        self, source_word: Optional[Union[str, int]], threshold: float = 0
    ) -> Iterable[Tuple[int, float]]:
        if source_word is None:
            source_word = 0
        elif isinstance(source_word, str):
            source_word = self._model.get_src_word_index(source_word)
        return self._model.get_translations(source_word, threshold)

    def _create_model(self) -> ta.AlignmentModel:
        return create_alignment_model(self.type)

    def _set_model(self, model: ta.AlignmentModel) -> None:
        self._model = model
        self._source_words = _ThotWordVocabulary(self._model, is_src=True)
        self._target_words = _ThotWordVocabulary(self._model, is_src=False)


class _ThotWordVocabulary(Sequence[str]):
    def __init__(self, model: ta.AlignmentModel, is_src: bool) -> None:
        self._model = model
        self._is_src = is_src

    def __getitem__(self, word_index: int) -> str:
        if word_index >= len(self):
            raise IndexError
        return self._model.get_src_word(word_index) if self._is_src else self._model.get_trg_word(word_index)

    def __len__(self) -> int:
        return self._model.src_vocab_size if self._is_src else self._model.trg_vocab_size

    def __contains__(self, x: object) -> bool:
        return any(self[i] == x for i in range(len(self)))

    def __iter__(self) -> Iterator[str]:
        return (self[i] for i in range(len(self)))

    def __reversed__(self) -> Iterator[str]:
        return (self[i] for i in reversed(range(len(self))))


def _to_word_alignment_matrix(alignment: Sequence[int], src_length: int) -> WordAlignmentMatrix:
    matrix = WordAlignmentMatrix(src_length, len(alignment))
    for j in range(len(alignment)):
        if alignment[j] > 0:
            matrix[alignment[j] - 1, j] = True
    return matrix


class _Trainer(ThotWordAlignmentModelTrainer):
    def __init__(
        self,
        model: ThotWordAlignmentModel,
        prefix_filename: Optional[StrPath],
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        corpus: ParallelTextCorpus,
        max_corpus_count: int,
    ) -> None:
        super().__init__(
            model.type,
            prefix_filename,
            source_preprocessor,
            target_preprocessor,
            corpus,
            max_corpus_count,
        )
        self._machine_model = model

    def save(self) -> None:
        super().save()
        self._machine_model._set_model(self._model)
