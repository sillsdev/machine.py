from abc import abstractmethod
from typing import Collection, Dict, Iterable, Optional, Sequence, Tuple, Union

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.corpus import Corpus
from ..corpora.parallel_text_row import ParallelTextRow
from .trainer import Trainer
from .word_aligner import WordAligner
from .word_alignment_matrix import WordAlignmentMatrix
from .word_vocabulary import WordVocabulary


class WordAlignmentModel(WordAligner):
    @property
    @abstractmethod
    def source_words(self) -> WordVocabulary:
        ...

    @property
    @abstractmethod
    def target_words(self) -> WordVocabulary:
        ...

    @property
    @abstractmethod
    def special_symbol_indices(self) -> Collection[int]:
        ...

    @abstractmethod
    def create_trainer(self, corpus: Corpus[ParallelTextRow]) -> Trainer:
        ...

    @abstractmethod
    def get_translations(
        self, source_word: Optional[Union[str, int]], threshold: float = 0
    ) -> Iterable[Tuple[int, float]]:
        ...

    @abstractmethod
    def get_translation_score(
        self, source_word: Optional[Union[str, int]], target_word: Optional[Union[str, int]]
    ) -> float:
        ...

    @abstractmethod
    def get_alignment_score(
        self,
        source_length: int,
        prev_source_index: int,
        source_index: int,
        target_length: int,
        prev_target_index: int,
        target_index: int,
    ) -> float:
        ...

    def get_translation_table(self, threshold: float = 0) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        source_words = list(self.source_words)
        target_words = list(self.target_words)
        for i in range(len(source_words)):
            row: Dict[str, float] = {}
            for j, score in self.get_translations(i, threshold):
                row[target_words[j]] = score
            results[source_words[i]] = row
        return results

    def get_aligned_word_pairs(
        self, source_segment: Sequence[str], target_segment: Sequence[str], wa_matrix: WordAlignmentMatrix
    ) -> Collection[AlignedWordPair]:
        word_pairs, source_indices, target_indices = wa_matrix.get_asymmetric_alignments()
        for word_pair in word_pairs:
            source_word = source_segment[word_pair.source_index]
            target_word = target_segment[word_pair.target_index]
            word_pair.translation_score = self.get_translation_score(source_word, target_word)

            prev_source_index = -1 if word_pair.target_index == 0 else source_indices[word_pair.target_index - 1]
            prev_target_index = -1 if word_pair.source_index == 0 else target_indices[word_pair.source_index - 1]
            word_pair.alignment_score = self.get_alignment_score(
                len(source_segment),
                prev_source_index,
                word_pair.source_index,
                len(target_segment),
                prev_target_index,
                word_pair.target_index,
            )
        return word_pairs

    def get_alignment_string(
        self,
        row: ParallelTextRow,
        include_scores: bool = True,
    ) -> str:
        alignment = self.get_best_alignment_from_known(
            row.source_segment, row.target_segment, WordAlignmentMatrix.from_parallel_text_corpus_row(row)
        )
        if not include_scores:
            return str(alignment)
        return " ".join(
            str(wp) for wp in self.get_aligned_word_pairs(row.source_segment, row.target_segment, alignment)
        )

    def get_giza_format_string(
        self,
        row: ParallelTextRow,
    ) -> str:
        alignment = self.get_best_alignment_from_known(
            row.source_segment, row.target_segment, WordAlignmentMatrix.from_parallel_text_corpus_row(row)
        )
        return alignment.to_giza_format(row.source_segment, row.target_segment)
