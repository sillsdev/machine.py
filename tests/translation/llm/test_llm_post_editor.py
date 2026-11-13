import json

import pytest
from testutils.mock_example_retriever import MockExampleRetriever
from testutils.mock_llm_completion_client import MockLlmCompletionClient

from machine.translation.llm import (
    Completion,
    CompletionFinishReason,
    ContentBlockError,
    ContextVerse,
    Example,
    LlmPostEditor,
    PostEditSegment,
    TfidfExampleRetriever,
)

_EXAMPLES = [
    Example("the cat", "el gato"),
    Example("the house", "la casa"),
    Example("the dog sleeps", "el perro duerme"),
]


def _create_answer(
    translation: str = "el perro", changed: bool = True, justification: str = "fixed the article's gender"
) -> Completion:
    text = json.dumps({"changed": changed, "justification": justification, "translation": translation})
    return Completion(text, 0.0, finish_reason=CompletionFinishReason.STOP)


def _create_post_editor(client: MockLlmCompletionClient, **kwargs) -> LlmPostEditor:
    return LlmPostEditor(client, MockExampleRetriever(_EXAMPLES), "English", "Spanish", **kwargs)


def _get_prompt(client: MockLlmCompletionClient, request: int = 0) -> str:
    return client.requests[request]["messages"][1]["content"]


def test_post_edit() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert result.changed
    assert result.justification == "fixed the article's gender"
    assert result.finish_reason is CompletionFinishReason.STOP


@pytest.mark.parametrize("source", ["", "   "])
def test_post_edit_empty_source(source: str) -> None:
    client = MockLlmCompletionClient([_create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment(source, "la perro"))

    assert result.translation == ""
    assert result.finish_reason is None
    assert result.total_cost == 0.0
    assert client.requests == []


def test_post_edit_totals_across_attempts() -> None:
    answer = _create_answer()
    client = MockLlmCompletionClient(
        [
            Completion("not json", 0.5, input_token_count=10, output_token_count=2),
            Completion(answer.text, 0.25, input_token_count=11, output_token_count=3),
        ]
    )

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert (result.total_cost, result.total_input_token_count, result.total_output_token_count) == (0.75, 21, 5)


def test_post_edit_default_num_examples() -> None:
    client = MockLlmCompletionClient([_create_answer()])
    examples = [Example(f"source {i:02d}", f"target {i:02d}") for i in range(60)]
    post_editor = LlmPostEditor(client, MockExampleRetriever(examples), "English", "Spanish")

    post_editor.post_edit(PostEditSegment("the dog", "la perro"))

    assert _get_prompt(client).count('<target lang="Spanish">') == 50


def test_post_edit_examples_least_similar_first() -> None:
    client = MockLlmCompletionClient([_create_answer()])
    examples = [
        Example("the dog", "el perro"),
        Example("the dog sleeps", "el perro duerme"),
        Example("a red house", "una casa roja"),
    ]
    post_editor = LlmPostEditor(client, TfidfExampleRetriever(examples), "English", "Spanish", num_examples=2)

    post_editor.post_edit(PostEditSegment("the dog", "la perro"))

    prompt = _get_prompt(client)
    assert "una casa roja" not in prompt
    assert prompt.index("el perro duerme") < prompt.index('<target lang="Spanish">el perro</target>')


def test_post_edit_draft() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert '<draft_translation lang="Spanish">\n  la perro\n</draft_translation>' in _get_prompt(client)


def test_post_edit_missing_draft() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "   "))

    assert '<draft_translation lang="Spanish" status="not_available"/>' in _get_prompt(client)


def test_post_edit_context_verse_offsets() -> None:
    client = MockLlmCompletionClient([_create_answer()])
    context = (ContextVerse(-1, "before", "antes"), ContextVerse(1, "after", "después"))

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro", context))

    prompt = _get_prompt(client)
    assert '<verse offset="-1">\n    <source lang="English">before</source>' in prompt
    assert '<target lang="Spanish">antes</target>' in prompt
    assert '<verse offset="1">\n    <source lang="English">after</source>' in prompt
    assert '<target lang="Spanish">después</target>' in prompt
    assert '<verse offset="0" role="to_translate">' in prompt
    assert prompt.index('offset="-1"') < prompt.index('offset="0"') < prompt.index('offset="1"')


def test_post_edit_untranslated_context_verse() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro", (ContextVerse(-1, "before"),)))

    assert '<target lang="Spanish" status="not_available"/>' in _get_prompt(client)


def test_post_edit_missing_context_verse() -> None:
    client = MockLlmCompletionClient([_create_answer()])
    context = (ContextVerse(-2, "two before", "dos antes"), ContextVerse(1, "after", "después"))

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro", context))

    assert '<missing_verse offset="-1"/>' in _get_prompt(client)


@pytest.mark.parametrize("source", ["", "   "])
def test_post_edit_empty_source_context_verse(source: str) -> None:
    client = MockLlmCompletionClient([_create_answer()])
    context = (ContextVerse(-2, "two before", "dos antes"), ContextVerse(-1, source, "antes"))

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro", context))

    prompt = _get_prompt(client)
    assert '<missing_verse offset="-1"/>' in prompt
    assert '<target lang="Spanish">antes</target>' not in prompt


def test_post_edit_empty_source_context_verse_at_edge() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro", (ContextVerse(1, ""),)))

    assert 'offset="1"' not in _get_prompt(client)


def test_post_edit_system_instruction_languages() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert "from English to Spanish" in client.requests[0]["messages"][0]["content"]


def test_post_edit_response_schema() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    schema = client.requests[0]["response_schema"]
    assert list(schema["properties"]) == ["changed", "justification", "translation"]
    assert schema["required"] == ["changed", "justification", "translation"]


def test_post_edit_retry_unparseable_answer() -> None:
    client = MockLlmCompletionClient([Completion("Sure! Here you go: ...", 0.0), _create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert len(client.requests) == 2
    assert "could not be parsed" not in _get_prompt(client, 0)
    assert _get_prompt(client, 1).startswith(_get_prompt(client, 0))
    assert "could not be parsed" in _get_prompt(client, 1)


def test_post_edit_retry_empty_translation() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="   ", changed=False), _create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert len(client.requests) == 2


def test_post_edit_max_attempts() -> None:
    unparseable = Completion("not json", 0.25, finish_reason=CompletionFinishReason.STOP)
    client = MockLlmCompletionClient([unparseable, unparseable])

    result = _create_post_editor(client, max_attempts=2).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == ""
    assert result.total_cost == 0.5
    assert result.finish_reason is CompletionFinishReason.STOP
    assert len(client.requests) == 2


def test_post_edit_max_attempts_zero() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        _create_post_editor(MockLlmCompletionClient([_create_answer()]), max_attempts=0)


def test_post_edit_retry_malformed_response() -> None:
    client = MockLlmCompletionClient([ValueError("Malformed response"), _create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert len(client.requests) == 2


def test_post_edit_content_block_not_retried() -> None:
    client = MockLlmCompletionClient([ContentBlockError([])])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == ""
    assert result.finish_reason is CompletionFinishReason.CONTENT_FILTER
    assert len(client.requests) == 1


def test_post_edit_newline_in_translation() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el perro\nduerme")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog sleeps", "la perro duerme"))

    assert result.translation == "el perro duerme"


def test_post_edit_batch() -> None:
    client = MockLlmCompletionClient([_create_answer(), _create_answer(translation="la casa")])

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "el casa")]
    )

    assert [result.translation for result in results] == ["el perro", "la casa"]
