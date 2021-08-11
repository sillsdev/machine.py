import xml.etree.ElementTree as etree
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Generator, List, Optional, Sequence, Set, Tuple

from ..annotations.range import Range
from ..scripture.verse_ref import VerseRef, Versification
from ..tokenization import RangeTokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from ..utils.typeshed import StrPath
from .aligned_word_pair import AlignedWordPair
from .corpora_helpers import get_scripture_text_sort_key, get_usx_id
from .text_alignment import TextAlignment
from .text_alignment_collection import TextAlignmentCollection
from .usx_token import UsxToken
from .usx_verse_parser import UsxVerseParser


class UsxFileTextAlignmentCollection(TextAlignmentCollection):
    def __init__(
        self,
        src_word_tokenizer: RangeTokenizer[str, int, str],
        trg_word_tokenizer: RangeTokenizer[str, int, str],
        src_filename: StrPath,
        trg_filename: StrPath,
        src_versification: Optional[Versification] = None,
        trg_versification: Optional[Versification] = None,
    ) -> None:
        self._src_word_tokenizer = src_word_tokenizer
        self._trg_word_tokenizer = trg_word_tokenizer
        self._src_filename = Path(src_filename)
        self._trg_filename = Path(trg_filename)
        self._src_versification = src_versification
        self._trg_versification = trg_versification

        self._id = get_usx_id(self._src_filename)
        self._sort_key = get_scripture_text_sort_key(self._id)

        self._parser = UsxVerseParser()

    @property
    def id(self) -> str:
        return self._id

    @property
    def sort_key(self) -> str:
        return self._sort_key

    @property
    def alignments(self) -> ContextManagedGenerator[TextAlignment, None, None]:
        return ContextManagedGenerator(self._get_alignments())

    def _get_alignments(self) -> Generator[TextAlignment, None, None]:
        with open(self._src_filename, "rb") as src_stream, open(self._trg_filename, "rb") as trg_stream:
            src_iterator = iter(self._parser.parse(src_stream))
            trg_iterator = iter(self._parser.parse(trg_stream))
            range_info = _RangeInfo()

            src_verse = next(src_iterator, None)
            trg_verse = next(trg_iterator, None)

            while src_verse is not None and trg_verse is not None:
                src_verse_ref = VerseRef(self.id, src_verse.chapter, src_verse.verse, self._src_versification)
                trg_verse_ref = VerseRef(self.id, trg_verse.chapter, trg_verse.verse, self._trg_versification)

                compare = src_verse_ref.compare_to(trg_verse_ref)
                if compare < 0:
                    src_verse = next(src_iterator, None)
                elif compare > 0:
                    trg_verse = next(trg_iterator, None)
                else:
                    if src_verse_ref.has_multiple or trg_verse_ref.has_multiple:
                        if range_info.verse_ref is not None and (
                            (src_verse_ref.has_multiple and not trg_verse_ref.has_multiple and len(src_verse.text) > 0)
                            or (
                                not src_verse_ref.has_multiple
                                and trg_verse_ref.has_multiple
                                and len(trg_verse.text) > 0
                            )
                            or (
                                src_verse_ref.has_multiple
                                and trg_verse_ref.has_multiple
                                and len(src_verse.text) > 0
                                and len(trg_verse.text) > 0
                            )
                        ):
                            range_alignment = self._create_text_alignment(
                                range_info.verse_ref, range_info.source_tokens, range_info.target_tokens
                            )
                            if len(range_alignment.aligned_word_pairs) > 0:
                                yield range_alignment
                        if range_info.verse_ref is None:
                            range_info.verse_ref = src_verse_ref
                        range_info.source_tokens.extend(src_verse.tokens)
                        range_info.target_tokens.extend(trg_verse.tokens)
                    else:
                        if range_info.verse_ref is not None:
                            range_alignment = self._create_text_alignment(
                                range_info.verse_ref, range_info.source_tokens, range_info.target_tokens
                            )
                            if len(range_alignment.aligned_word_pairs) > 0:
                                yield range_alignment

                        alignment = self._create_text_alignment(src_verse_ref, src_verse.tokens, trg_verse.tokens)
                        if len(alignment.aligned_word_pairs) > 0:
                            yield alignment
                    src_verse = next(src_iterator, None)
                    trg_verse = next(trg_iterator, None)

    def invert(self) -> "UsxFileTextAlignmentCollection":
        return UsxFileTextAlignmentCollection(
            self._trg_word_tokenizer,
            self._src_word_tokenizer,
            self._trg_filename,
            self._src_filename,
            self._trg_versification,
            self._src_versification,
        )

    def _create_text_alignment(
        self, verse_ref: VerseRef, src_tokens: Sequence[UsxToken], trg_tokens: Sequence[UsxToken]
    ) -> TextAlignment:
        src_links = _get_links(self._src_word_tokenizer, src_tokens)
        trg_links = _get_links(self._trg_word_tokenizer, trg_tokens)

        word_pairs: Set[AlignedWordPair] = set()
        for link_id, src_indices in src_links.items():
            trg_indices = trg_links.get(link_id)
            if trg_indices is not None:
                for src_index in src_indices:
                    for trg_index in trg_indices:
                        word_pairs.add(AlignedWordPair(src_index, trg_index))
        return TextAlignment(self.id, verse_ref, word_pairs)


@dataclass
class _RangeInfo:
    verse_ref: Optional[VerseRef] = None
    source_tokens: List[UsxToken] = field(default_factory=list)
    target_tokens: List[UsxToken] = field(default_factory=list)


def _get_links(word_tokenizer: RangeTokenizer[str, int, str], tokens: Sequence[UsxToken]) -> DefaultDict[str, Set[int]]:
    prev_para_elem: Optional[etree.Element] = None
    text = ""
    link_strs: List[Tuple[Range[int], str]] = []
    for token in tokens:
        if token.para_element != prev_para_elem and len(text) > 0:
            text += " "

        start = len(text)
        text += str(token)
        if token.element is not None and token.element.tag == "wg":
            link_strs.append((Range.create(start, len(text)), token.element.get("target_links", "")))
        prev_para_elem = token.para_element
    text = text.strip()

    i = 0
    segment_links: DefaultDict[str, Set[int]] = defaultdict(lambda: set())
    token_iterator = iter(word_tokenizer.tokenize_as_ranges(text))
    link_str_iterator = iter(link_strs)
    token_range = next(token_iterator, None)
    link_str_tuple = next(link_str_iterator, None)
    while token_range is not None and link_str_tuple is not None:
        link_range, link_str = link_str_tuple
        links = link_str.split(";")

        compare = token_range.compare_to(link_range)
        if compare < 0:
            if token_range.contains(link_range):
                for link in links:
                    segment_links[link].add(i)
            else:
                token_range = next(token_iterator, None)
                i += 1
        elif compare > 0:
            if link_range.contains(token_range):
                for link in links:
                    segment_links[link].add(i)
            else:
                link_str_tuple = next(link_str_iterator, None)
        else:
            for link in links:
                segment_links[link].add(i)

            token_range = next(token_iterator, None)
            i += 1
            link_str_tuple = next(link_str_iterator, None)
    return segment_links
