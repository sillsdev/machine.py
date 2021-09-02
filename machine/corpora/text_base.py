from abc import abstractmethod
from typing import Any, Generator

from ..tokenization import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .text import Text
from .text_segment import TextSegment
from .token_processors import UNESCAPE_SPACES


class TextBase(Text):
    def __init__(self, word_tokenizer: Tokenizer[str, int, str], id: str, sort_key: str) -> None:
        self._word_tokenizer = word_tokenizer
        self._id = id
        self._sort_key = sort_key

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    @property
    def word_tokenizer(self) -> Tokenizer[str, int, str]:
        return self._word_tokenizer

    def get_segments(self, include_text: bool = True) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(self._get_segments(include_text))

    @abstractmethod
    def _get_segments(self, include_text: bool) -> Generator[TextSegment, None, None]:
        ...

    def _create_text_segment(
        self,
        include_text: bool,
        text: str,
        seg_ref: Any,
        is_sentence_start: bool = True,
        is_in_range: bool = False,
        is_range_start: bool = False,
    ) -> TextSegment:
        text = text.strip()
        if not include_text:
            return TextSegment(
                self._id, seg_ref, [], is_sentence_start, is_in_range, is_range_start, is_empty=len(text) == 0
            )
        segment = list(self._word_tokenizer.tokenize(text))
        segment = UNESCAPE_SPACES.process(segment)
        return TextSegment(
            self._id, seg_ref, segment, is_sentence_start, is_in_range, is_range_start, is_empty=len(segment) == 0
        )

    def _create_empty_text_segment(self, seg_ref: Any, is_in_range: bool = False) -> TextSegment:
        return TextSegment(
            self._id, seg_ref, [], is_sentence_start=True, is_in_range=is_in_range, is_range_start=False, is_empty=True
        )
