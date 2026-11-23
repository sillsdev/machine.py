from pathlib import Path
from typing import Any, List

from transformers import Trainer

from machine.translation.translation_engine import TranslationEngine

from ..corpora.parallel_text_corpus import ParallelTextCorpus
from ..corpora.text_corpus import TextCorpus
from ..translation.llm.example import Example
from ..translation.llm.example_retriever_factory import ExampleRetrieverFactory
from ..translation.llm.example_retriever_method import ExampleRetrievalMethod
from ..translation.llm.llm_completion_client import LlmCompletionClient, LlmUsageTotals
from ..translation.llm.remote_llm_engine import RemoteLlmEngine
from .nmt_model_factory import NmtModelFactory


class RemoteLlmModelFactory(NmtModelFactory):

    def __init__(
        self,
        config: Any,
        completion_client: LlmCompletionClient,
        retrieval_method: ExampleRetrievalMethod,
        usage_totals: LlmUsageTotals,
    ) -> None:
        self._config = config
        self._num_examples = config.remote_llm.num_examples
        self._batch_size = config.remote_llm.batch_size
        self._num_workers = config.remote_llm.num_workers
        self._completion_client = completion_client
        self._example_retrieval_method = retrieval_method
        self._usage_totals = usage_totals

    def create_engine(self) -> TranslationEngine:
        return RemoteLlmEngine(
            self._completion_client,
            self._example_retriever,
            self._config.src_lang,
            self._config.trg_lang,
            self._num_examples,
            self._batch_size,
            self._num_workers,
            self._usage_totals,
        )

    def init_example_retriever(self, corpus: ParallelTextCorpus) -> None:
        examples: List[Example] = []
        with corpus.get_rows() as rows:
            for row in rows:
                if not row.is_empty:
                    examples.append(Example(row.source_text, row.target_text))
        self._example_retriever = ExampleRetrieverFactory.create_example_retriever(
            self._example_retrieval_method, examples
        )

    @property
    def train_tokenizer(self) -> bool:
        return False

    def init(self) -> None:
        return

    def create_source_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        raise TypeError("Tokenizer training is not available for remote LLM models.")

    def create_target_tokenizer_trainer(self, corpus: TextCorpus) -> Trainer:
        raise TypeError("Tokenizer training is not available for remote LLM models.")

    def create_model_trainer(self, corpus: ParallelTextCorpus) -> Trainer:
        raise TypeError("Model training is not available for remote LLM models.")

    def save_model(self) -> Path:
        raise TypeError("Model saving is not available for remote LLM models.")
