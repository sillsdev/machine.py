from __future__ import annotations

from typing import Optional, Sequence, Union, cast

from datasets.arrow_dataset import Dataset
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.typeshed import StrPath
from ..translation_model import TranslationModel
from ..translation_result import TranslationResult
from .hugging_face_nmt_engine import HuggingFaceNmtEngine
from .hugging_face_nmt_model_trainer import HuggingFaceNmtModelTrainer


class HuggingFaceNmtModel(TranslationModel):
    def __init__(
        self,
        model_path: StrPath,
        parent_model_name: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        batch_size: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        self._model_path = model_path
        self._parent_model_name = parent_model_name
        if training_args is None:
            parser = HfArgumentParser(Seq2SeqTrainingArguments)
            training_args = cast(Seq2SeqTrainingArguments, parser.parse_dict({"output_dir": str(self._model_path)})[0])
        self._training_args = training_args
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._batch_size = batch_size
        self._device = device
        self._engine: Optional[HuggingFaceNmtEngine] = None

    @property
    def parent_model_name(self) -> str:
        return self._parent_model_name

    @property
    def source_lang(self) -> Optional[str]:
        return self._source_lang

    @property
    def target_lang(self) -> Optional[str]:
        return self._target_lang

    @property
    def training_args(self) -> Seq2SeqTrainingArguments:
        return self._training_args

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        return self._get_engine().translate(segment)

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        return self._get_engine().translate_batch(segments)

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Sequence[TranslationResult]:
        return self._get_engine().translate_n(n, segment)

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        return self._get_engine().translate_n_batch(n, segments)

    def create_trainer(self, corpus: Union[ParallelTextCorpus, Dataset]) -> HuggingFaceNmtModelTrainer:
        return _Trainer(self, corpus)

    def __enter__(self) -> HuggingFaceNmtModel:
        return self

    def reset_engine(self) -> None:
        self._engine = None

    def _get_engine(self) -> HuggingFaceNmtEngine:
        if self._engine is None:
            self._engine = HuggingFaceNmtEngine(
                self._model_path, self._source_lang, self._target_lang, self._batch_size, self._device
            )
        return self._engine


class _Trainer(HuggingFaceNmtModelTrainer):
    def __init__(self, model: HuggingFaceNmtModel, corpus: Union[ParallelTextCorpus, Dataset]) -> None:
        self._model = model
        super().__init__(model.parent_model_name, model.training_args, corpus, model._source_lang, model._target_lang)

    def save(self) -> None:
        super().save()
        self._model.reset_engine()
