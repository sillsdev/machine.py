from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from datasets.arrow_dataset import Dataset
from transformers import PreTrainedModel, Seq2SeqTrainingArguments

from ...corpora.parallel_text_corpus import ParallelTextCorpus
from ...utils.typeshed import StrPath
from ..translation_model import TranslationModel
from ..translation_result import TranslationResult
from .hugging_face_nmt_engine import HuggingFaceNmtEngine
from .hugging_face_nmt_model_trainer import HuggingFaceNmtModelTrainer


class HuggingFaceNmtModel(TranslationModel):
    def __init__(
        self,
        model: Union[PreTrainedModel, StrPath],
        parent_model_name: str,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        **pipeline_kwargs,
    ) -> None:
        self._model = model
        if isinstance(model, PreTrainedModel):
            self._model_path = Path(model.name_or_path)
        else:
            self._model_path = Path(model)
        self._parent_model_name = parent_model_name
        if training_args is None:
            training_args = Seq2SeqTrainingArguments(output_dir=str(self._model_path))
        self._training_args = training_args
        self._pipeline_kwargs = pipeline_kwargs
        self._engine: Optional[HuggingFaceNmtEngine] = None

    @property
    def parent_model_name(self) -> str:
        return self._parent_model_name

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

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.reset_engine()

    def reset_engine(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def _get_engine(self) -> HuggingFaceNmtEngine:
        if self._engine is None:
            self._engine = HuggingFaceNmtEngine(self._model, **self._pipeline_kwargs)
        return self._engine


class _Trainer(HuggingFaceNmtModelTrainer):
    def __init__(self, model: HuggingFaceNmtModel, corpus: Union[ParallelTextCorpus, Dataset]) -> None:
        self._model = model
        src_lang = model._pipeline_kwargs.get("src_lang")
        tgt_lang = model._pipeline_kwargs.get("tgt_lang")
        max_length = model._pipeline_kwargs.get("max_length")
        super().__init__(
            model.parent_model_name, model.training_args, corpus, src_lang, tgt_lang, max_length, max_length
        )

    def save(self) -> None:
        super().save()
        output_dir = Path(self._model.training_args.output_dir)
        if output_dir != self._model._model_path:
            shutil.copytree(output_dir, self._model._model_path)
        self._model.reset_engine()
