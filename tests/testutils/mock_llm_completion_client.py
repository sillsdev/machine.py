from typing import Any, Dict, List, Optional, Sequence, Union

from machine.translation.llm.llm_completion_client import Completion, LlmCompletionClient


class MockLlmCompletionClient(LlmCompletionClient):
    """Returns the queued completions in order, raising any queued exception, and records what it was asked for.

    Raises IndexError when called more times than it has completions, so an unexpected extra call fails the test.
    """

    def __init__(self, completions: Sequence[Union[Completion, Exception]]) -> None:
        self._completions = list(completions)
        self.requests: List[Dict[str, Any]] = []

    def complete(
        self, messages: List[Dict[str, str]], logprobs: bool = False, response_schema: Optional[Dict[str, Any]] = None
    ) -> Completion:
        self.requests.append({"messages": messages, "logprobs": logprobs, "response_schema": response_schema})
        completion = self._completions[len(self.requests) - 1]
        if isinstance(completion, Exception):
            raise completion
        return completion
