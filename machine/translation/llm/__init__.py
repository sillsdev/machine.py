from ...utils.packages import is_sklearn_available

if not is_sklearn_available():
    raise RuntimeError('sklearn is not installed. Install sil-machine with the "llm" extra.')

from .completion_finish_reason import CompletionFinishReason
from .context_verse import ContextVerse
from .coverage_example_retriever import CoverageExampleRetriever
from .example import Example
from .example_retriever import ExampleRetriever
from .example_retriever_factory import ExampleRetrieverFactory
from .example_retriever_method import ExampleRetrievalMethod
from .llm_completion_client import (
    Completion,
    CompletionError,
    ContentBlockError,
    LlmCompletionClient,
    LlmUsageSummary,
    LlmUsageTotals,
    MalformedCompletionError,
    RefusalError,
    TokenLogprob,
    TruncatedCompletionError,
)
from .llm_post_editor import LlmPostEditor
from .openrouter_llm_completion_client import OpenRouterLlmCompletionClient
from .post_edit_batch_error import PostEditBatchError
from .post_edit_result import PostEditResult
from .post_edit_segment import PostEditSegment
from .remote_llm_engine import RemoteLlmEngine
from .tfidf_example_retriever import TfidfExampleRetriever
from .top_k_example_retriever import TopKExampleRetriever

__all__ = [
    "CoverageExampleRetriever",
    "Example",
    "ExampleRetrieverFactory",
    "ExampleRetrievalMethod",
    "ExampleRetriever",
    "ContentBlockError",
    "Completion",
    "CompletionError",
    "CompletionFinishReason",
    "ContextVerse",
    "LlmCompletionClient",
    "LlmPostEditor",
    "MalformedCompletionError",
    "LlmUsageSummary",
    "LlmUsageTotals",
    "OpenRouterLlmCompletionClient",
    "PostEditBatchError",
    "PostEditResult",
    "PostEditSegment",
    "RemoteLlmEngine",
    "RefusalError",
    "TokenLogprob",
    "TfidfExampleRetriever",
    "TopKExampleRetriever",
    "TruncatedCompletionError",
]
