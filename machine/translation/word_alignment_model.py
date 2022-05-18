from abc import abstractmethod
from statistics import mean
from typing import Collection, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

from ..corpora.aligned_word_pair import AlignedWordPair
from ..corpora.parallel_text_corpus import ParallelTextCorpus
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
    def create_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
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

    def get_best_aligned_word_pairs(
        self, source_segment: Sequence[str], target_segment: Sequence[str]
    ) -> Collection[AlignedWordPair]:
        wa_matrix = self.get_best_alignment(source_segment, target_segment)
        word_pairs = wa_matrix.to_aligned_word_pairs()
        self.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)
        return word_pairs

    def get_best_aligned_word_pairs_batch(
        self, segments: Iterable[Tuple[Sequence[str], Sequence[str]]]
    ) -> Iterable[Tuple[Sequence[str], Sequence[str], Collection[AlignedWordPair]]]:
        results = self.get_best_alignment_batch(segments)
        for source_segment, target_segment, wa_matrix in results:
            word_pairs = wa_matrix.to_aligned_word_pairs()
            self.compute_aligned_word_pair_scores(source_segment, target_segment, word_pairs)
            yield source_segment, target_segment, word_pairs

    def compute_aligned_word_pair_scores(
        self, source_segment: Sequence[str], target_segment: Sequence[str], word_pairs: Collection[AlignedWordPair]
    ) -> None:
        alignment_score = 1.0 / (len(source_segment) + 1)
        for word_pair in word_pairs:
            source_word = None if word_pair.source_index == -1 else source_segment[word_pair.source_index]
            target_word = None if word_pair.target_index == -1 else target_segment[word_pair.target_index]
            word_pair.translation_score = self.get_translation_score(source_word, target_word)
            word_pair.alignment_score = alignment_score

    def get_avg_translation_score(
        self, source_segment: Sequence[str], target_segment: Sequence[str], wa_matrix: WordAlignmentMatrix
    ) -> float:
        scores: List[float] = []
        for word_pair in wa_matrix.to_aligned_word_pairs(include_null=True):
            source_word = None if word_pair.source_index == -1 else source_segment[word_pair.source_index]
            target_word = None if word_pair.target_index == -1 else target_segment[word_pair.target_index]
            scores.append(self.get_translation_score(source_word, target_word))
        return mean(scores) if len(scores) > 0 else 0

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
        word_pairs = alignment.to_aligned_word_pairs()
        self.compute_aligned_word_pair_scores(row.source_segment, row.target_segment, word_pairs)
        return " ".join(str(wp) for wp in word_pairs)

    def get_giza_format_string(
        self,
        row: ParallelTextRow,
    ) -> str:
        alignment = self.get_best_alignment_from_known(
            row.source_segment, row.target_segment, WordAlignmentMatrix.from_parallel_text_corpus_row(row)
        )
        return alignment.to_giza_format(row.source_segment, row.target_segment)


def align_corpus(model: WordAlignmentModel, corpus: ParallelTextCorpus, batch_size: int = 1024) -> ParallelTextCorpus:
    return _AlignParallelTextCorpus(corpus, model, batch_size)


class _AlignParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, model: WordAlignmentModel, batch_size: int) -> None:
        self._corpus = corpus
        self._model = model
        self._batch_size = batch_size

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.batch(self._batch_size) as batches:
            for row_batch in batches:
                alignments = self._model.get_best_alignment_batch(
                    (r.source_segment, r.target_segment) for r in row_batch
                )
                for row, (_, _, alignment) in zip(row_batch, alignments):
                    known_alignment = WordAlignmentMatrix.from_parallel_text_corpus_row(row)
                    if known_alignment is not None:
                        known_alignment.priority_symmetrize_with(alignment)
                        alignment = known_alignment
                    word_pairs = alignment.to_aligned_word_pairs()
                    self._model.compute_aligned_word_pair_scores(row.source_segment, row.target_segment, word_pairs)
                    row.aligned_word_pairs = word_pairs
                    yield row
