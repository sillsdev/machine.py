from typing import Generator, List, Sequence

from ..annotations.range import Range
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.range_tokenizer import RangeTokenizer
from ..tokenization.tokenization_utils import get_ranges, split
from .error_correction_model import ErrorCorrectionModel
from .error_correction_word_graph_processor import ErrorCorrectionWordGraphProcessor
from .interactive_translation_engine import InteractiveTranslationEngine
from .translation_constants import MAX_SEGMENT_LENGTH
from .translation_result import TranslationResult
from .word_graph import WordGraph


class InteractiveTranslator:
    def __init__(
        self,
        ecm: ErrorCorrectionModel,
        engine: InteractiveTranslationEngine,
        target_tokenizer: RangeTokenizer[str, int, str],
        target_detokenizer: Detokenizer[str, str],
        segment: str,
        word_graph: WordGraph,
        sentence_start: bool,
    ) -> None:
        self._segment = segment
        self._segment_word_ranges = list(get_ranges(self._segment, word_graph.source_tokens))
        self._engine = engine
        self._target_tokenizer = target_tokenizer
        self._prefix_word_ranges: List[Range[int]] = []
        self._prefix = ""
        self._is_last_word_complete = True
        self._word_graph_processor = ErrorCorrectionWordGraphProcessor(ecm, target_detokenizer, word_graph)
        self._target_detokenizer = target_detokenizer
        self._sentence_start = sentence_start
        self._correct()

    @property
    def target_detokenizer(self) -> Detokenizer[str, str]:
        return self._target_detokenizer

    @property
    def segment(self) -> str:
        return self._segment

    @property
    def segment_word_ranges(self) -> Sequence[Range[int]]:
        return self._segment_word_ranges

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def prefix_word_ranges(self) -> Sequence[Range[int]]:
        return self._prefix_word_ranges

    @property
    def is_last_word_complete(self) -> bool:
        return self._is_last_word_complete

    @property
    def sentence_start(self) -> bool:
        return self._sentence_start

    @property
    def is_segment_valid(self) -> bool:
        return len(self.segment_word_ranges) <= MAX_SEGMENT_LENGTH

    def set_prefix(self, prefix: str) -> None:
        if self._prefix != prefix:
            self._prefix = prefix
            self._correct()

    def append_to_prefix(self, addition: str) -> None:
        if addition != "":
            self._prefix += addition
            self._correct()

    def approve(self, aligned_only: bool) -> None:
        if not self.is_segment_valid or len(self.prefix_word_ranges) > MAX_SEGMENT_LENGTH:
            return

        segment_word_ranges = self._segment_word_ranges
        if aligned_only:
            best_result = next(self.get_current_results(), None)
            if best_result is None:
                return
            segment_word_ranges = self._get_aligned_source_segment(best_result)

        if len(segment_word_ranges) > 0:
            source_segment = self._segment[segment_word_ranges[0].start : segment_word_ranges[-1].end]
            target_segment = self._prefix[self._prefix_word_ranges[0].start : self._prefix_word_ranges[-1].end]
            self._engine.train_segment(source_segment, target_segment, self._sentence_start)

    def get_current_results(self) -> Generator[TranslationResult, None, None]:
        return self._word_graph_processor.get_results()

    def _correct(self) -> None:
        self._prefix_word_ranges = list(self._target_tokenizer.tokenize_as_ranges(self._prefix))
        self._is_last_word_complete = len(self._prefix_word_ranges) == 0 or self._prefix_word_ranges[-1].end < len(
            self._prefix
        )
        self._word_graph_processor.correct(split(self._prefix, self._prefix_word_ranges), self._is_last_word_complete)

    def _get_aligned_source_segment(self, result: TranslationResult) -> Sequence[Range[int]]:
        source_length = 0
        for phrase in result.phrases:
            if phrase.target_segment_cut > len(self._prefix_word_ranges):
                break
            if phrase.source_segment_range.end > source_length:
                source_length = phrase.source_segment_range.end

        return (
            self._segment_word_ranges
            if source_length == len(self._segment_word_ranges)
            else self._segment_word_ranges[:source_length]
        )
