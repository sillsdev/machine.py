from __future__ import annotations

from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from ..scripture.verse_ref import VerseRef
from ..tokenization.detokenizer import Detokenizer
from ..tokenization.tokenizer import Tokenizer
from ..utils.context_managed_generator import ContextManagedGenerator
from .aligned_word_pair import AlignedWordPair
from .corpora_utils import get_split_indices
from .corpus import Corpus
from .parallel_text_row import ParallelTextRow
from .token_processors import escape_spaces, lowercase, normalize, unescape_spaces

if TYPE_CHECKING:
    import pandas as pd
    from datasets.arrow_dataset import Dataset
    from datasets.info import DatasetInfo
    from datasets.iterable_dataset import IterableDataset
    from datasets.splits import NamedSplit


class ParallelTextCorpus(Corpus[ParallelTextRow]):
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        text_id_column: Optional[str] = "text",
        ref_column: Optional[str] = "ref",
        source_column: str = "source",
        target_column: str = "target",
        alignment_column: Optional[str] = "alignment",
        default_text_id: str = "*all*",
        ref_factory: Optional[Callable[[Any], Any]] = None,
    ) -> ParallelTextCorpus:
        return _PandasParallelTextCorpus(
            df,
            text_id_column,
            ref_column,
            source_column,
            target_column,
            alignment_column,
            default_text_id,
            ref_factory,
        )

    @classmethod
    def from_hf_dataset(
        cls,
        ds: Union[Dataset, IterableDataset],
        source_lang: str,
        target_lang: str,
        text_id_column: Optional[str] = "text",
        ref_column: Optional[str] = "ref",
        translation_column: str = "translation",
        alignment_column: Optional[str] = "alignment",
        default_text_id: str = "*all*",
        ref_factory: Optional[Callable[[Any], Any]] = None,
    ) -> ParallelTextCorpus:
        return _DatasetParallelTextCorpus(
            ds,
            source_lang,
            target_lang,
            text_id_column,
            ref_column,
            translation_column,
            alignment_column,
            default_text_id,
            ref_factory,
        )

    def invert(self) -> ParallelTextCorpus:
        def _invert(row: ParallelTextRow) -> ParallelTextRow:
            return row.invert()

        return self.transform(_invert)

    def tokenize(
        self, source_tokenizer: Tokenizer[str, int, str], target_tokenizer: Optional[Tokenizer[str, int, str]] = None
    ) -> ParallelTextCorpus:
        if target_tokenizer is None:
            target_tokenizer = source_tokenizer

        def _tokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.source_segment) > 0:
                row.source_segment = list(source_tokenizer.tokenize(row.source_text))
            if len(row.target_segment) > 0:
                row.target_segment = list(target_tokenizer.tokenize(row.target_text))
            return row

        return self.transform(_tokenize)

    def tokenize_source(self, tokenizer: Tokenizer[str, int, str]) -> ParallelTextCorpus:
        def _tokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.source_segment) > 0:
                row.source_segment = list(tokenizer.tokenize(row.source_text))
            return row

        return self.transform(_tokenize)

    def tokenize_target(self, tokenizer: Tokenizer[str, int, str]) -> ParallelTextCorpus:
        def _tokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.target_segment) > 0:
                row.target_segment = list(tokenizer.tokenize(row.target_text))
            return row

        return self.transform(_tokenize)

    def detokenize(
        self, source_detokenizer: Detokenizer[str, str], target_detokenizer: Optional[Detokenizer[str, str]] = None
    ) -> ParallelTextCorpus:
        if target_detokenizer is None:
            target_detokenizer = source_detokenizer

        def _detokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.source_segment) > 1:
                row.source_segment = [source_detokenizer.detokenize(row.source_segment)]
            if len(row.target_segment) > 1:
                row.target_segment = [target_detokenizer.detokenize(row.target_segment)]
            return row

        return self.transform(_detokenize)

    def detokenize_source(self, detokenizer: Detokenizer[str, str]) -> ParallelTextCorpus:
        def _detokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.source_segment) > 1:
                row.source_segment = [detokenizer.detokenize(row.source_segment)]
            return row

        return self.transform(_detokenize)

    def detokenize_target(self, detokenizer: Detokenizer[str, str]) -> ParallelTextCorpus:
        def _detokenize(row: ParallelTextRow) -> ParallelTextRow:
            if len(row.target_segment) > 1:
                row.target_segment = [detokenizer.detokenize(row.target_segment)]
            return row

        return self.transform(_detokenize)

    def normalize(self, normalization_form: str) -> ParallelTextCorpus:
        def _normalize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = normalize(normalization_form, row.source_segment)
            row.target_segment = normalize(normalization_form, row.target_segment)
            return row

        return self.transform(_normalize)

    def normalize_source(self, normalization_form: str) -> ParallelTextCorpus:
        def _normalize(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = normalize(normalization_form, row.source_segment)
            return row

        return self.transform(_normalize)

    def normalize_target(self, normalization_form: str) -> ParallelTextCorpus:
        def _normalize(row: ParallelTextRow) -> ParallelTextRow:
            row.target_segment = normalize(normalization_form, row.target_segment)
            return row

        return self.transform(_normalize)

    def nfc_normalize(self) -> ParallelTextCorpus:
        return self.normalize("NFC")

    def nfc_normalize_source(self) -> ParallelTextCorpus:
        return self.normalize_source("NFC")

    def nfc_normalize_target(self) -> ParallelTextCorpus:
        return self.normalize_target("NFC")

    def nfd_normalize(self) -> ParallelTextCorpus:
        return self.normalize("NFD")

    def nfd_normalize_source(self) -> ParallelTextCorpus:
        return self.normalize_source("NFD")

    def nfd_normalize_target(self) -> ParallelTextCorpus:
        return self.normalize_target("NFD")

    def nfkc_normalize(self) -> ParallelTextCorpus:
        return self.normalize("NFKC")

    def nfkc_normalize_source(self) -> ParallelTextCorpus:
        return self.normalize_source("NFKC")

    def nfkc_normalize_target(self) -> ParallelTextCorpus:
        return self.normalize_target("NFKC")

    def nfkd_normalize(self) -> ParallelTextCorpus:
        return self.normalize("NFKD")

    def nfkd_normalize_source(self) -> ParallelTextCorpus:
        return self.normalize_source("NFKD")

    def nfkd_normalize_target(self) -> ParallelTextCorpus:
        return self.normalize_target("NFKD")

    def lowercase(self) -> ParallelTextCorpus:
        def _lowercase(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = lowercase(row.source_segment)
            row.target_segment = lowercase(row.target_segment)
            return row

        return self.transform(_lowercase)

    def lowercase_source(self) -> ParallelTextCorpus:
        def _lowercase(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = lowercase(row.source_segment)
            return row

        return self.transform(_lowercase)

    def lowercase_target(self) -> ParallelTextCorpus:
        def _lowercase(row: ParallelTextRow) -> ParallelTextRow:
            row.target_segment = lowercase(row.target_segment)
            return row

        return self.transform(_lowercase)

    def escape_spaces(self) -> ParallelTextCorpus:
        def _escape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = escape_spaces(row.source_segment)
            row.target_segment = escape_spaces(row.target_segment)
            return row

        return self.transform(_escape_spaces)

    def unescape_spaces(self) -> ParallelTextCorpus:
        def _unescape_spaces(row: ParallelTextRow) -> ParallelTextRow:
            row.source_segment = unescape_spaces(row.source_segment)
            row.target_segment = unescape_spaces(row.target_segment)
            return row

        return self.transform(_unescape_spaces)

    def transform(self, transform: Callable[[ParallelTextRow], ParallelTextRow]) -> ParallelTextCorpus:
        return _TransformParallelTextCorpus(self, transform)

    def filter_nonempty(self) -> ParallelTextCorpus:
        return self.filter(lambda r: not r.is_empty)

    def filter(self, predicate: Callable[[ParallelTextRow], bool]) -> ParallelTextCorpus:
        return self.filter_by_index(lambda r, _: predicate(r))

    def filter_by_index(self, predicate: Callable[[ParallelTextRow, int], bool]) -> ParallelTextCorpus:
        return _FilterParallelTextCorpus(self, predicate)

    def take(self, count: int) -> ParallelTextCorpus:
        return _TakeParallelTextCorpus(self, count)

    def split(
        self, percent: Optional[float] = None, size: Optional[int] = None, include_empty: bool = True, seed: Any = None
    ) -> Tuple[ParallelTextCorpus, ParallelTextCorpus, int, int]:
        corpus_size = self.count(include_empty)
        split_indices = get_split_indices(corpus_size, percent, size, seed)

        main_corpus = self.filter_by_index(lambda r, i: i not in split_indices and (include_empty or not r.is_empty))
        split_corpus = self.filter_by_index(lambda r, i: i in split_indices and (include_empty or not r.is_empty))

        return main_corpus, split_corpus, corpus_size - len(split_indices), len(split_indices)

    def to_tuples(self) -> ContextManagedGenerator[Tuple[Sequence[str], Sequence[str]], None, None]:
        return self.map(lambda r: (r.source_segment, r.target_segment))

    def to_pandas(
        self,
        text_id_column: Optional[str] = "text",
        ref_column: Optional[str] = "ref",
        source_column: Optional[str] = "source",
        target_column: Optional[str] = "target",
        alignment_column: Optional[str] = "alignment",
    ) -> pd.DataFrame:
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas is not installed.")

        text_ids: Optional[List[str]] = None if text_id_column is None else []
        refs: Optional[List[Any]] = None if ref_column is None else []
        source: Optional[List[str]] = None if source_column is None else []
        target: Optional[List[str]] = None if target_column is None else []
        alignments: Optional[List[List[Tuple[int, int]]]] = None if alignment_column is None else []
        has_alignments = False
        with self.get_rows() as rows:
            for row in rows:
                if text_ids is not None:
                    text_ids.append(row.text_id)
                if refs is not None:
                    refs.append(row.ref)
                if source is not None:
                    source.append(row.source_text)
                if target is not None:
                    target.append(row.target_text)
                if alignments is not None:
                    if row.aligned_word_pairs is not None:
                        alignment = [(r.source_index, r.target_index) for r in row.aligned_word_pairs]
                        has_alignments = True
                    else:
                        alignment = []
                    alignments.append(alignment)

        data = {}
        if text_ids is not None:
            data[text_id_column] = text_ids
        if refs is not None:
            data[ref_column] = refs
        if source is not None:
            data[source_column] = source
        if target is not None:
            data[target_column] = target
        if alignments is not None and has_alignments:
            data[alignment_column] = alignments
        return pd.DataFrame(data)

    def to_hf_dataset(
        self,
        source_lang: str,
        target_lang: str,
        text_id_column: Optional[str] = "text",
        ref_column: Optional[str] = "ref",
        translation_column: str = "translation",
        alignment_column: Optional[str] = "alignment",
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> IterableDataset:
        try:
            from datasets.features.features import Features, FeatureType, Sequence, Value
            from datasets.features.translation import Translation
            from datasets.info import DatasetInfo
            from datasets.iterable_dataset import ExamplesIterable, IterableDataset
        except ImportError:
            raise RuntimeError("datasets is not installed.")

        features_dict: Dict[str, FeatureType] = {translation_column: Translation(languages=[source_lang, target_lang])}
        if text_id_column is not None:
            features_dict[text_id_column] = Value("string")
        if ref_column is not None:
            features_dict[ref_column] = Sequence(Value("string"))
        if alignment_column is not None:
            features_dict[alignment_column] = Sequence({source_lang: Value("int32"), target_lang: Value("int32")})
        features = Features(features_dict)

        if info is not None and info.features is not None and info.features != features:
            raise ValueError("Features specified in `info.features` is not compatible.")

        if info is None:
            info = DatasetInfo()

        info.features = features

        def iterable() -> Iterable[Tuple[Union[str, int], dict]]:
            with self.get_rows() as rows:
                for row in rows:
                    key = row.ref
                    if not isinstance(key, int) and not isinstance(key, str):
                        key = str(key)
                    example = {}
                    if text_id_column is not None:
                        example[text_id_column] = row.text_id
                    if ref_column is not None:
                        example[ref_column] = row.refs
                    example[translation_column] = {source_lang: row.source_text, target_lang: row.target_text}
                    if alignment_column is not None and row.aligned_word_pairs is not None:
                        src_indices: List[int] = []
                        trg_indices: List[int] = []
                        for wp in row.aligned_word_pairs:
                            src_indices.append(wp.source_index)
                            trg_indices.append(wp.target_index)
                        example[alignment_column] = {source_lang: src_indices, target_lang: trg_indices}
                    yield key, example

        return IterableDataset(ExamplesIterable(iterable, {}), info, split)


def flatten_parallel_text_corpora(corpora: Iterable[ParallelTextCorpus]) -> ParallelTextCorpus:
    corpus_list = list(corpora)
    if len(corpus_list) == 1:
        return corpus_list[0]

    return _FlattenParallelTextCorpus(corpus_list)


class _TransformParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, transform: Callable[[ParallelTextRow], ParallelTextRow]):
        self._corpus = corpus
        self._transform = transform

    @property
    def missing_rows_allowed(self) -> bool:
        return self._corpus.missing_rows_allowed

    def count(self, include_empty: bool = True) -> int:
        return self._corpus.count(include_empty)

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from map(self._transform, rows)


class _FilterParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, predicate: Callable[[ParallelTextRow, int], bool]) -> None:
        self._corpus = corpus
        self._predicate = predicate

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from (row for i, row in enumerate(rows) if self._predicate(row, i))


class _TakeParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpus: ParallelTextCorpus, count: int) -> None:
        self._corpus = corpus
        self._count = count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        with self._corpus.get_rows() as rows:
            yield from islice(rows, self._count)


class _FlattenParallelTextCorpus(ParallelTextCorpus):
    def __init__(self, corpora: List[ParallelTextCorpus]) -> None:
        self._corpora = corpora

    @property
    def missing_rows_allowed(self) -> bool:
        return any(corpus.missing_rows_allowed for corpus in self._corpora)

    def count(self, include_empty: bool = True) -> int:
        return sum(corpus.count(include_empty) for corpus in self._corpora)

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        for corpus in self._corpora:
            with corpus.get_rows() as rows:
                yield from rows


class _PandasParallelTextCorpus(ParallelTextCorpus):
    def __init__(
        self,
        df: pd.DataFrame,
        text_id_column: Optional[str],
        ref_column: Optional[str],
        source_column: str,
        target_column: str,
        alignment_column: Optional[str],
        default_text_id: str,
        ref_factory: Optional[Callable[[Any], Any]],
    ) -> None:
        self._df = df
        self._text_id_column = text_id_column
        self._ref_column = ref_column
        self._source_column = source_column
        self._target_column = target_column
        self._alignment_column = alignment_column
        self._default_text_id = default_text_id
        self._ref_factory = ref_factory

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        if include_empty:
            return len(self._df)
        return len(self._df[(self._df[self._source_column] != "") & (self._df[self._target_column] != "")])

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        for index, row in self._df.iterrows():
            text_id = self._default_text_id
            if self._text_id_column is not None and self._text_id_column in self._df:
                text_id = cast(str, row[self._text_id_column])
            refs = [index]
            if self._ref_column is not None and self._ref_column in self._df:
                refs = row[self._ref_column]
                if not isinstance(refs, list):
                    refs = [refs]
                if self._ref_factory is not None:
                    refs = [self._ref_factory(ref) for ref in refs]
            source = cast(str, row[self._source_column])
            target = cast(str, row[self._target_column])
            alignment: Optional[Collection[AlignedWordPair]] = None
            if self._alignment_column is not None and self._alignment_column in self._df:
                v = row[self._alignment_column]
                alignment = (
                    AlignedWordPair.from_string(v) if isinstance(v, str) else [AlignedWordPair(t[0], t[1]) for t in v]
                )
            yield ParallelTextRow(
                text_id, refs, refs, [source] if len(source) > 0 else [], [target] if len(target) > 0 else [], alignment
            )


class _DatasetParallelTextCorpus(ParallelTextCorpus):
    def __init__(
        self,
        ds: Union[Dataset, IterableDataset],
        source_lang: str,
        target_lang: str,
        text_id_column: Optional[str],
        ref_column: Optional[str],
        translation_column: str,
        alignment_column: Optional[str],
        default_text_id: str,
        ref_factory: Optional[Callable[[Any], Any]],
    ) -> None:
        self._ds = ds
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._text_id_column = text_id_column
        self._ref_column = ref_column
        self._translation_column = translation_column
        self._alignment_column = alignment_column
        self._default_text_id = default_text_id
        self._ref_factory = ref_factory

    @property
    def missing_rows_allowed(self) -> bool:
        return False

    def count(self, include_empty: bool = True) -> int:
        try:
            from datasets.arrow_dataset import Dataset
        except ImportError:
            raise RuntimeError("datasets is not installed.")
        if include_empty and isinstance(self._ds, Dataset):
            return len(self._ds)

        count = 0
        for example in cast(Iterable[dict], self._ds):
            if (
                self._get_translation(self._source_lang, example) != ""
                and self._get_translation(self._target_lang, example) != ""
            ):
                count += 1
        return count

    def _get_rows(self) -> Generator[ParallelTextRow, None, None]:
        index = 0
        for example in cast(Iterable[dict], self._ds):
            text_id = self._default_text_id
            refs = [index]
            if self._ref_column is not None and self._ref_column in example:
                refs = example[self._ref_column]
                if not isinstance(refs, list):
                    refs = [refs]
                if self._ref_factory is not None:
                    refs = [self._ref_factory(ref) for ref in refs]
                    if len(refs) > 0 and isinstance(refs[0], VerseRef):
                        text_id = refs[0].book
            if self._text_id_column is not None and self._text_id_column in example:
                text_id = cast(str, example[self._text_id_column])
            source = self._get_translation(self._source_lang, example)
            target = self._get_translation(self._target_lang, example)
            alignment: Optional[Collection[AlignedWordPair]] = None
            if self._alignment_column is not None and self._alignment_column in example:
                src_indices = example[self._alignment_column][self._source_lang]
                trg_indices = example[self._alignment_column][self._target_lang]
                alignment = [AlignedWordPair(si, ti) for (si, ti) in zip(src_indices, trg_indices)]

            yield ParallelTextRow(
                text_id, refs, refs, [source] if len(source) > 0 else [], [target] if len(target) > 0 else [], alignment
            )
            index += 1

    def _get_translation(self, lang: str, example: dict) -> str:
        translation: dict = example[self._translation_column]
        if lang in translation:
            return translation[lang]
        if "language" in translation and "translation" in translation:
            languages: List[str] = translation["language"]
            translations: List[str] = translation["translation"]
            try:
                index = languages.index(lang)
                return translations[index]
            except ValueError:
                return ""
        return ""
