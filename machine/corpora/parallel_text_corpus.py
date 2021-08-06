from typing import Generator, Iterable, Optional, Set

from ..utils.context_managed_generator import ContextManagedGenerator
from .dictionary_text_alignment_corpus import DictionaryTextAlignmentCorpus
from .null_text import NullText
from .null_text_alignment_collection import NullTextAlignmentCollection
from .parallel_text import ParallelText
from .parallel_text_segment import ParallelTextSegment
from .text import Text
from .text_alignment_collection import TextAlignmentCollection
from .text_alignment_corpus import TextAlignmentCorpus
from .text_corpus import TextCorpus
from .text_segment import TextSegment


def _get_text(corpus: TextCorpus, id: str) -> Text:
    text = corpus.get_text(id)
    if text is None:
        text = NullText(id, corpus.get_text_sort_key(id))
    return text


def _get_text_alignment_collection(corpus: TextAlignmentCorpus, id: str) -> TextAlignmentCollection:
    alignments = corpus.get_text_alignment_collection(id)
    if alignments is None:
        alignments = NullTextAlignmentCollection(id, corpus.get_text_alignment_collection_sort_key(id))
    return alignments


class ParallelTextCorpus:
    def __init__(
        self,
        source_corpus: TextCorpus,
        target_corpus: TextCorpus,
        text_alignment_corpus: Optional[TextAlignmentCorpus] = None,
    ) -> None:
        self._source_corpus = source_corpus
        self._target_corpus = target_corpus
        self._text_alignment_corpus = (
            DictionaryTextAlignmentCorpus() if text_alignment_corpus is None else text_alignment_corpus
        )

    @property
    def source_corpus(self) -> TextCorpus:
        return self._source_corpus

    @property
    def target_corpus(self) -> TextCorpus:
        return self._target_corpus

    @property
    def text_alignment_corpus(self) -> TextAlignmentCorpus:
        return self._text_alignment_corpus

    @property
    def texts(self) -> Iterable[ParallelText]:
        return self.get_texts()

    @property
    def segments(self) -> ContextManagedGenerator[ParallelTextSegment, None, None]:
        return self.get_segments()

    @property
    def source_segments(self) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(self._get_source_segments())

    @property
    def target_segments(self) -> ContextManagedGenerator[TextSegment, None, None]:
        return ContextManagedGenerator(self._get_target_segments())

    def invert(self) -> "ParallelTextCorpus":
        return ParallelTextCorpus(
            self._target_corpus,
            self._source_corpus,
            None if self._text_alignment_corpus is None else self._text_alignment_corpus.invert(),
        )

    def get_texts(self, all_source_segments: bool = False, all_target_segments: bool = False) -> Iterable[ParallelText]:
        source_text_ids = {t.id for t in self._source_corpus.texts}
        target_text_ids = {t.id for t in self._target_corpus.texts}

        text_ids: Set[str]
        if all_source_segments and all_target_segments:
            text_ids = source_text_ids | target_text_ids
        elif not all_source_segments and not all_target_segments:
            text_ids = source_text_ids & target_text_ids
        elif all_source_segments:
            text_ids = source_text_ids
        else:
            text_ids = target_text_ids

        return sorted((self._create_parallel_text(id) for id in text_ids), key=lambda t: t.sort_key)

    def get_segments(
        self, all_source_segments: bool = False, all_target_segments: bool = False, include_text: bool = True
    ) -> ContextManagedGenerator[ParallelTextSegment, None, None]:
        return ContextManagedGenerator(self._get_segments(all_source_segments, all_target_segments, include_text))

    def get_count(
        self, all_source_segments: bool = False, all_target_segments: bool = False, nonempty_only: bool = False
    ) -> int:
        return sum(
            t.get_count(all_source_segments, all_target_segments, nonempty_only)
            for t in self.get_texts(all_source_segments, all_target_segments)
        )

    def _get_segments(
        self, all_source_segments: bool, all_target_segments: bool, include_text: bool
    ) -> Generator[ParallelTextSegment, None, None]:
        for text in self.get_texts(all_source_segments, all_target_segments):
            with text.get_segments(all_source_segments, all_target_segments, include_text) as segments:
                for segment in segments:
                    yield segment

    def _get_source_segments(self) -> Generator[TextSegment, None, None]:
        for text in self.texts:
            with text.source_text.get_segments() as segments:
                for segment in segments:
                    yield segment

    def _get_target_segments(self) -> Generator[TextSegment, None, None]:
        for text in self.texts:
            with text.target_text.get_segments() as segments:
                for segment in segments:
                    yield segment

    def _create_parallel_text(self, id: str) -> ParallelText:
        source_text = _get_text(self._source_corpus, id)
        target_text = _get_text(self._target_corpus, id)
        text_alignment_collection = _get_text_alignment_collection(self._text_alignment_corpus, id)
        return ParallelText(source_text, target_text, text_alignment_collection)
