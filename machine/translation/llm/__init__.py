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
from .llm_completion_client import Completion, LlmCompletionClient, TokenLogprob
from .llm_post_editor import LlmPostEditor
from .openrouter_llm_completion_client import ContentBlockError, OpenRouterLlmCompletionClient
from .post_edit_result import PostEditResult
from .post_edit_segment import PostEditSegment
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
    "CompletionFinishReason",
    "ContextVerse",
    "LlmCompletionClient",
    "LlmPostEditor",
    "OpenRouterLlmCompletionClient",
    "PostEditResult",
    "PostEditSegment",
    "TokenLogprob",
    "TfidfExampleRetriever",
    "TopKExampleRetriever",
]
