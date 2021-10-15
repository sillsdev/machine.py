import sys
from abc import abstractmethod
from pathlib import Path
from typing import Collection, Iterable, Iterator, Optional, Sequence, Tuple, Union

import thot.alignment as ta

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...corpora.token_processors import NO_OP, TokenProcessor
from ...utils.typeshed import StrPath
from ..ibm1_word_alignment_model import Ibm1WordAlignmentModel
from ..trainer import Trainer
from ..word_alignment_matrix import WordAlignmentMatrix
from ..word_vocabulary import WordVocabulary
from .thot_word_alignment_model_trainer import ThotWordAlignmentModelTrainer
from .thot_word_alignment_model_type import ThotWordAlignmentModelType
from .thot_word_alignment_parameters import ThotWordAlignmentParameters

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
        self.parameters = ThotWordAlignmentParameters()

    @property
    def source_words(self) -> WordVocabulary:
        return self._source_words

    @property
    def target_words(self) -> WordVocabulary:
        return self._target_words

    @property
    def special_symbol_indices(self) -> Collection[int]:
        return _SPECIAL_SYMBOL_INDICES

    @property
    def thot_model(self) -> ta.AlignmentModel:
        return self._model

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
        corpus: ParallelTextCorpus,
        source_preprocessor: TokenProcessor = NO_OP,
        target_preprocessor: TokenProcessor = NO_OP,
        max_corpus_count: int = sys.maxsize,
    ) -> Trainer:
        return _Trainer(self, corpus, self._prefix_filename, source_preprocessor, target_preprocessor, max_corpus_count)

    def get_best_alignment(self, source_segment: Sequence[str], target_segment: Sequence[str]) -> WordAlignmentMatrix:
        _, matrix = self._model.get_best_alignment(source_segment, target_segment)
        return WordAlignmentMatrix(matrix.to_numpy())

    def get_best_alignments(
        self, source_segments: Sequence[Sequence[str]], target_segments: Sequence[Sequence[str]]
    ) -> Sequence[WordAlignmentMatrix]:
        if len(source_segments) != len(target_segments):
            raise ValueError("The number of source and target segments must be equal.")
        return [
            WordAlignmentMatrix(matrix.to_numpy())
            for _, matrix in self._model.get_best_alignments(source_segments, target_segments)
        ]

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
        return self._model.translation_prob(source_word, target_word)

    def get_translations(
        self, source_word: Optional[Union[str, int]], threshold: float = 0
    ) -> Iterable[Tuple[int, float]]:
        if source_word is None:
            source_word = 0
        elif isinstance(source_word, str):
            source_word = self._model.get_src_word_index(source_word)
        return self._model.get_translations(source_word, threshold)

    def _create_model(self) -> ta.AlignmentModel:
        if self.type is ThotWordAlignmentModelType.IBM1:
            return ta.Ibm1AlignmentModel()
        elif self.type is ThotWordAlignmentModelType.IBM2:
            return ta.Ibm2AlignmentModel()
        elif self.type is ThotWordAlignmentModelType.IBM3:
            return ta.Ibm3AlignmentModel()
        elif self.type is ThotWordAlignmentModelType.IBM4:
            return ta.Ibm4AlignmentModel()
        elif self.type is ThotWordAlignmentModelType.HMM:
            return ta.HmmAlignmentModel()
        elif self.type is ThotWordAlignmentModelType.FAST_ALIGN:
            return ta.FastAlignModel()
        else:
            raise ValueError("The model type is invalid.")

    def _set_model(self, model: ta.AlignmentModel) -> None:
        self._model = model
        self._source_words = _ThotWordVocabulary(self._model, is_src=True)
        self._target_words = _ThotWordVocabulary(self._model, is_src=False)


class _ThotWordVocabulary(WordVocabulary):
    def __init__(self, model: ta.AlignmentModel, is_src: bool) -> None:
        self._model = model
        self._is_src = is_src

    def index(self, word: Optional[str]) -> int:
        if word is None:
            return 0
        return self._model.get_src_word_index(word) if self._is_src else self._model.get_trg_word_index(word)

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


class _Trainer(ThotWordAlignmentModelTrainer):
    def __init__(
        self,
        model: ThotWordAlignmentModel,
        corpus: ParallelTextCorpus,
        prefix_filename: Optional[StrPath],
        source_preprocessor: TokenProcessor,
        target_preprocessor: TokenProcessor,
        max_corpus_count: int,
    ) -> None:
        super().__init__(
            model.type,
            corpus,
            prefix_filename,
            model.parameters,
            source_preprocessor,
            target_preprocessor,
            max_corpus_count,
        )
        self._machine_model = model

    def save(self) -> None:
        super().save()
        self._machine_model._set_model(self._model)
