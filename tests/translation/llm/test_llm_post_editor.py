import json
import threading
from typing import Any, Dict, List, Optional

import pytest
from testutils.mock_example_retriever import MockExampleRetriever
from testutils.mock_llm_completion_client import MockLlmCompletionClient

from machine.translation.llm import (
    Completion,
    CompletionFinishReason,
    ContentBlockError,
    ContextVerse,
    Example,
    LlmCompletionClient,
    LlmPostEditor,
    MalformedCompletionError,
    PostEditBatchError,
    PostEditSegment,
    RefusalError,
    TfidfExampleRetriever,
    TruncatedCompletionError,
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


def _create_post_editor(client: LlmCompletionClient, **kwargs) -> LlmPostEditor:
    return LlmPostEditor(client, MockExampleRetriever(_EXAMPLES), "English", "Spanish", **kwargs)


def _get_prompt(client: MockLlmCompletionClient, request: int = 0) -> str:
    return client.requests[request]["messages"][1]["content"]


def test_post_edit() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    result = _create_post_editor(client, include_justification=True).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert result.succeeded
    assert result.changed
    assert result.justification == "fixed the article's gender"
    assert result.finish_reason is CompletionFinishReason.STOP


def test_post_edit_no_change() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el perro", changed=False, justification="")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "el perro"))

    assert result.translation == "el perro"
    assert result.succeeded
    assert not result.changed


def test_post_edit_changed_reported_false() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el perro", changed=False, justification="")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.changed


def test_post_edit_unchanged_reported_true() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el perro", changed=True)])

    result = _create_post_editor(client, include_justification=True).post_edit(PostEditSegment("the dog", "el perro"))

    assert not result.changed
    assert result.justification == ""


def test_post_edit_unchanged_draft_line_break() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el perro", changed=False, justification="")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "el perro\n"))

    assert not result.changed


def test_post_edit_changed_only_whitespace_within_line() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="¿el perro ?", changed=False, justification="")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "¿el perro ?"))

    assert result.changed


@pytest.mark.parametrize("source", ["", "   "])
def test_post_edit_empty_source(source: str) -> None:
    client = MockLlmCompletionClient([_create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment(source, "la perro"))

    assert result.translation == "la perro"
    assert not result.succeeded
    assert not result.changed
    assert result.finish_reason is None
    assert result.total_cost == 0.0
    assert client.requests == []


def test_post_edit_totals_across_attempts() -> None:
    answer = _create_answer()
    client = MockLlmCompletionClient(
        [
            MalformedCompletionError([], "not json", cost=0.5, input_token_count=10, output_token_count=2),
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


def test_post_edit_prompt_escapes_markup() -> None:
    client = MockLlmCompletionClient([_create_answer()])
    retriever = MockExampleRetriever([Example("cats & dogs", "gatos & perros")])
    post_editor = LlmPostEditor(client, retriever, 'English "US"', "Spanish")
    context = (ContextVerse(-1, "a < b", "a > b"),)

    post_editor.post_edit(PostEditSegment("the </source> dog", "la & perro", context))

    prompt = _get_prompt(client)
    assert "<source lang='English \"US\"'>cats &amp; dogs</source>" in prompt
    assert '<target lang="Spanish">gatos &amp; perros</target>' in prompt
    assert "a &lt; b</source>" in prompt
    assert '<target lang="Spanish">a &gt; b</target>' in prompt
    assert ">the &lt;/source&gt; dog</source>" in prompt
    assert '<draft_translation lang="Spanish">\n  la &amp; perro\n</draft_translation>' in prompt
    assert prompt.endswith(":\nthe &lt;/source&gt; dog")


def test_post_edit_answer_unescaped() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el &amp; perro", justification="&lt;fixed&gt;")])

    result = _create_post_editor(client, include_justification=True).post_edit(
        PostEditSegment("the & dog", "la & perro")
    )

    assert result.translation == "el & perro"
    assert result.justification == "<fixed>"


def test_post_edit_echoed_escaped_draft_unchanged() -> None:
    client = MockLlmCompletionClient([_create_answer(translation="el &amp; perro", changed=False, justification="")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the & dog", "el & perro"))

    assert result.translation == "el & perro"
    assert not result.changed


def test_post_edit_system_instruction_languages() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert "from English to Spanish" in client.requests[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    "include_justification,keys",
    [(False, ["translation"]), (True, ["changed", "justification", "translation"])],
)
def test_post_edit_response_schema(include_justification: bool, keys: list) -> None:
    client = MockLlmCompletionClient([_create_answer()])

    post_editor = _create_post_editor(client, include_justification=include_justification)
    post_editor.post_edit(PostEditSegment("the dog", "la perro"))

    schema = client.requests[0]["response_schema"]
    assert list(schema["properties"]) == keys
    assert schema["required"] == keys


@pytest.mark.parametrize(
    "include_justification,keys",
    [(False, '"translation"'), (True, '"changed", "justification", "translation"')],
)
def test_post_edit_retry_reminder_keys(include_justification: bool, keys: str) -> None:
    client = MockLlmCompletionClient([MalformedCompletionError([], "not json"), _create_answer()])

    post_editor = _create_post_editor(client, include_justification=include_justification)
    post_editor.post_edit(PostEditSegment("the dog", "la perro"))

    assert _get_prompt(client, 1).endswith(f"with exactly the keys {keys}.")


def test_post_edit_justification_omitted_by_default() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    system_instruction = client.requests[0]["messages"][0]["content"]
    assert "justification" not in system_instruction
    assert '"changed"' not in system_instruction
    assert result.changed
    assert result.justification == ""


def test_post_edit_justification_requested() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client, include_justification=True).post_edit(PostEditSegment("the dog", "la perro"))

    system_instruction = client.requests[0]["messages"][0]["content"]
    assert "providing your justification for the fix" in system_instruction
    assert '- "justification":' in system_instruction
    assert '- "changed":' in system_instruction


def test_post_edit_retry_unparseable_answer() -> None:
    client = MockLlmCompletionClient([MalformedCompletionError([], "not json"), _create_answer()])

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
    empty = Completion(_create_answer(translation="  ").text, 0.25, finish_reason=CompletionFinishReason.STOP)
    client = MockLlmCompletionClient([empty, empty])

    result = _create_post_editor(client, max_attempts=2).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "la perro"
    assert not result.succeeded
    assert not result.changed
    assert result.total_cost == 0.5
    assert result.finish_reason is CompletionFinishReason.STOP
    assert len(client.requests) == 2


def test_post_edit_failure_returns_single_line_draft() -> None:
    client = MockLlmCompletionClient([ContentBlockError([], "blocked")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog sleeps", " la perro\n duerme \n"))

    assert result.translation == "la perro duerme"


def test_post_edit_empty_source_returns_single_line_draft() -> None:
    client = MockLlmCompletionClient([])

    result = _create_post_editor(client).post_edit(PostEditSegment("  ", " la perro\n duerme \n"))

    assert result.translation == "la perro duerme"


def test_post_edit_prompt_source_stripped() -> None:
    client = MockLlmCompletionClient([_create_answer()])

    _create_post_editor(client).post_edit(PostEditSegment(" the dog\n", "la perro"))

    assert _get_prompt(client).endswith("Source (English):\nthe dog")


def test_post_edit_max_attempts_zero() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        _create_post_editor(MockLlmCompletionClient([_create_answer()]), max_attempts=0)


def test_post_edit_retry_malformed_response() -> None:
    client = MockLlmCompletionClient([MalformedCompletionError([], "no choices"), _create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "el perro"
    assert len(client.requests) == 2


def test_post_edit_invalid_request_not_retried() -> None:
    client = MockLlmCompletionClient([ValueError("temperature must be a number"), _create_answer()])

    with pytest.raises(ValueError, match="temperature"):
        _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert len(client.requests) == 1


@pytest.mark.parametrize("declined", [ContentBlockError, RefusalError, TruncatedCompletionError])
def test_post_edit_totals_include_failed_attempts(declined) -> None:
    client = MockLlmCompletionClient(
        [
            MalformedCompletionError([], "not json", cost=0.5, input_token_count=10, output_token_count=2),
            declined([], "declined", cost=0.25, input_token_count=11, output_token_count=3),
        ]
    )

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert not result.succeeded
    assert (result.total_cost, result.total_input_token_count, result.total_output_token_count) == (0.75, 21, 5)


def test_post_edit_content_block_not_retried() -> None:
    client = MockLlmCompletionClient([ContentBlockError([], "blocked")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "la perro"
    assert not result.succeeded
    assert not result.changed
    assert result.finish_reason is CompletionFinishReason.CONTENT_FILTER
    assert len(client.requests) == 1


def test_post_edit_refusal_not_retried() -> None:
    client = MockLlmCompletionClient([RefusalError([], "refused")])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "la perro"
    assert not result.succeeded
    assert not result.changed
    assert result.finish_reason is CompletionFinishReason.REFUSAL
    assert len(client.requests) == 1


def test_post_edit_truncation_not_retried() -> None:
    client = MockLlmCompletionClient([TruncatedCompletionError([], "cut off"), _create_answer()])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog", "la perro"))

    assert result.translation == "la perro"
    assert not result.succeeded
    assert not result.changed
    assert result.finish_reason is CompletionFinishReason.LENGTH
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "translation",
    ["el perro\nduerme", "el perro\r\nduerme", "el perro \n\n duerme", "el perro duerme", "\nel perro\nduerme\n"],
)
def test_post_edit_line_breaks_in_translation(translation: str) -> None:
    client = MockLlmCompletionClient([_create_answer(translation=translation)])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog sleeps", "la perro duerme"))

    assert result.translation == "el perro duerme"


@pytest.mark.parametrize("translation", ["¿el perro ?", "el perro duerme", "el  perro\tduerme"])
def test_post_edit_whitespace_within_line_preserved(translation: str) -> None:
    client = MockLlmCompletionClient([_create_answer(translation=translation)])

    result = _create_post_editor(client).post_edit(PostEditSegment("the dog sleeps", "la perro duerme"))

    assert result.translation == translation


def test_post_edit_batch() -> None:
    client = MockLlmCompletionClient([_create_answer(), _create_answer(translation="la casa")])

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "el casa")]
    )

    assert [result.translation for result in results] == ["el perro", "la casa"]


def test_post_edit_batch_keeps_results_after_failure() -> None:
    failure = RuntimeError("connection reset")
    client = MockLlmCompletionClient([_create_answer(), failure, _create_answer(translation="la casa")])
    segments = [
        PostEditSegment("the dog", "la perro"),
        PostEditSegment("the cat", " el gato\n negro "),
        PostEditSegment("the house", "el casa"),
    ]

    with pytest.raises(PostEditBatchError, match="1 of 3") as error:
        _create_post_editor(client).post_edit_batch(segments)

    assert [result.translation for result in error.value.results] == ["el perro", "el gato negro", "la casa"]
    assert [result.succeeded for result in error.value.results] == [True, False, True]
    assert error.value.errors == {1: failure}
    assert error.value.__cause__ is failure


class _UppercasingClient(LlmCompletionClient):
    """Answers each prompt with its source in upper case, failing for the source "fail".

    Each call waits at a barrier, so a test hangs into a BrokenBarrierError unless the calls overlap.
    """

    def __init__(self, parties: int) -> None:
        super().__init__()
        self._barrier = threading.Barrier(parties, timeout=5)

    def complete(
        self, messages: List[Dict[str, str]], logprobs: bool = False, response_schema: Optional[Dict[str, Any]] = None
    ) -> Completion:
        self._barrier.wait()
        source = messages[1]["content"].rsplit("\n", 1)[1]
        if source == "fail":
            raise RuntimeError("connection reset")
        return Completion(json.dumps({"translation": source.upper()}), 0.0, finish_reason=CompletionFinishReason.STOP)


def test_post_edit_batch_in_parallel_keeps_segment_order() -> None:
    sources = ["one", "two", "three", "four"]
    post_editor = _create_post_editor(_UppercasingClient(parties=4))

    results = post_editor.post_edit_batch([PostEditSegment(source, "draft") for source in sources], max_workers=4)

    assert [result.translation for result in results] == ["ONE", "TWO", "THREE", "FOUR"]


def test_post_edit_batch_in_parallel_keeps_results_after_failure() -> None:
    sources = ["one", "fail", "three"]
    post_editor = _create_post_editor(_UppercasingClient(parties=3))

    with pytest.raises(PostEditBatchError) as error:
        post_editor.post_edit_batch([PostEditSegment(source, "draft") for source in sources], max_workers=3)

    assert [result.translation for result in error.value.results] == ["ONE", "draft", "THREE"]
    assert list(error.value.errors) == [1]


def test_post_edit_batch_max_workers_zero() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        _create_post_editor(MockLlmCompletionClient([])).post_edit_batch([], max_workers=0)


def _create_passage_answer(
    *translations: str, cost: float = 0.0, input_token_count: int = 0, output_token_count: int = 0
) -> Completion:
    verses = [{"number": number, "translation": translation} for number, translation in enumerate(translations, 1)]
    return Completion(
        json.dumps({"verses": verses}),
        cost,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        finish_reason=CompletionFinishReason.STOP,
    )


def test_post_edit_batch_segments_per_request() -> None:
    client = MockLlmCompletionClient(
        [_create_passage_answer("el perro", "la casa"), _create_answer(translation="el gato")]
    )
    segments = [
        PostEditSegment("the dog", "la perro"),
        PostEditSegment("the house", "el casa"),
        PostEditSegment("the cat", "la gato"),
    ]

    results = _create_post_editor(client).post_edit_batch(segments, segments_per_request=2)

    assert [result.translation for result in results] == ["el perro", "la casa", "el gato"]
    assert all(result.succeeded and result.changed for result in results)
    assert results[0].finish_reason is CompletionFinishReason.STOP
    assert len(client.requests) == 2
    assert "verses" in client.requests[0]["response_schema"]["properties"]
    assert "translation" in client.requests[1]["response_schema"]["properties"]


def test_post_edit_batch_passage_prompt() -> None:
    client = MockLlmCompletionClient([_create_passage_answer("uno", "dos", "tres")])
    segments = [
        PostEditSegment("one", "un", (ContextVerse(-1, "zero", "cero"), ContextVerse(1, "stale two", "dos"))),
        PostEditSegment("two & <b>", "  "),
        PostEditSegment("three", "tres", (ContextVerse(-1, "stale two"), ContextVerse(2, "five", "cinco"))),
    ]

    _create_post_editor(client).post_edit_batch(segments, segments_per_request=3)

    prompt = _get_prompt(client)
    assert (
        "<passage_context>\n"
        '  <verse offset="-1">\n'
        '    <source lang="English">zero</source>\n'
        '    <target lang="Spanish">cero</target>\n'
        "  </verse>\n"
        '  <verse offset="0" role="to_translate" number="1">\n'
        '    <source lang="English">one</source>\n'
        "  </verse>\n"
        '  <verse offset="1" role="to_translate" number="2">\n'
        '    <source lang="English">two &amp; &lt;b&gt;</source>\n'
        "  </verse>\n"
        '  <verse offset="2" role="to_translate" number="3">\n'
        '    <source lang="English">three</source>\n'
        "  </verse>\n"
        '  <missing_verse offset="3"/>\n'
        '  <verse offset="4">\n'
        '    <source lang="English">five</source>\n'
        '    <target lang="Spanish">cinco</target>\n'
        "  </verse>\n"
        "</passage_context>"
    ) in prompt
    assert (
        '<draft_translation lang="Spanish">\n'
        '  <draft number="1">un</draft>\n'
        '  <draft number="2" status="not_available"/>\n'
        '  <draft number="3">tres</draft>\n'
        "</draft_translation>"
    ) in prompt
    assert prompt.endswith("Source (English):\n1. one\n2. two &amp; &lt;b&gt;\n3. three")
    system_instruction = client.requests[0]["messages"][0]["content"]
    assert "one entry per numbered verse" in system_instruction
    assert "Read the numbered verses together" in system_instruction


def test_post_edit_batch_passage_examples_for_whole_run() -> None:
    client = MockLlmCompletionClient([_create_passage_answer("el perro", "la casa")])
    examples = [Example("the dog", "el perro"), Example("the house", "la casa"), Example("a red car", "un coche rojo")]
    post_editor = LlmPostEditor(client, TfidfExampleRetriever(examples), "English", "Spanish", num_examples=2)

    post_editor.post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "el casa")], segments_per_request=2
    )

    prompt = _get_prompt(client)
    assert '<target lang="Spanish">el perro</target>' in prompt
    assert '<target lang="Spanish">la casa</target>' in prompt
    assert "un coche rojo" not in prompt


@pytest.mark.parametrize(
    "include_justification, keys",
    [(False, ["number", "translation"]), (True, ["number", "changed", "justification", "translation"])],
)
def test_post_edit_batch_passage_response_schema(include_justification: bool, keys: list) -> None:
    client = MockLlmCompletionClient([_create_passage_answer("el perro", "la casa")])

    _create_post_editor(client, include_justification=include_justification).post_edit_batch(
        [PostEditSegment("the dog", "el perro"), PostEditSegment("the house", "la casa")], segments_per_request=2
    )

    verse_schema = client.requests[0]["response_schema"]["properties"]["verses"]["items"]
    assert verse_schema["required"] == keys
    assert verse_schema["additionalProperties"] is False


def test_post_edit_batch_passage_justification() -> None:
    verses = [
        {"number": 1, "changed": True, "justification": "fixed the article", "translation": "el perro"},
        {"number": 2, "changed": False, "justification": "", "translation": "la casa"},
    ]
    client = MockLlmCompletionClient([Completion(json.dumps({"verses": verses}), 0.0)])

    results = _create_post_editor(client, include_justification=True).post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "la casa")], segments_per_request=2
    )

    assert [(result.changed, result.justification) for result in results] == [
        (True, "fixed the article"),
        (False, ""),
    ]


def test_post_edit_batch_passage_shares_usage() -> None:
    client = MockLlmCompletionClient(
        [_create_passage_answer("el perro", "la casa", cost=0.3, input_token_count=101, output_token_count=7)]
    )

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "el casa")], segments_per_request=2
    )

    assert [result.total_cost for result in results] == pytest.approx([0.15, 0.15])
    assert [(result.total_input_token_count, result.total_output_token_count) for result in results] == [
        (51, 4),
        (50, 3),
    ]


_FAILED_USAGE: Dict[str, Any] = {"cost": 0.5, "input_token_count": 10, "output_token_count": 2}


@pytest.mark.parametrize(
    "failure",
    [
        Completion(json.dumps({"verses": [{"number": 1, "translation": "el perro"}]}), **_FAILED_USAGE),
        Completion(
            json.dumps({"verses": [{"number": 2, "translation": "la casa"}, {"number": 1, "translation": "el perro"}]}),
            **_FAILED_USAGE,
        ),
        Completion(
            json.dumps({"verses": [{"number": 1, "translation": "el perro"}, {"number": 2, "translation": " "}]}),
            **_FAILED_USAGE,
        ),
        MalformedCompletionError([], "not json", **_FAILED_USAGE),
        TruncatedCompletionError([], "cut off", **_FAILED_USAGE),
        ContentBlockError([], "blocked", **_FAILED_USAGE),
    ],
    ids=["too few", "out of order", "empty translation", "malformed", "truncated", "blocked"],
)
def test_post_edit_batch_passage_unusable_reply_halved(failure) -> None:
    single_answer = Completion(
        _create_answer(translation="la casa").text, 0.25, input_token_count=20, output_token_count=4
    )
    client = MockLlmCompletionClient([failure, _create_answer(), single_answer])

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment("the dog", "la perro"), PostEditSegment("the house", "el casa")], segments_per_request=2
    )

    assert [result.translation for result in results] == ["el perro", "la casa"]
    assert len(client.requests) == 3
    assert [result.total_cost for result in results] == pytest.approx([0.25, 0.5])
    assert [(result.total_input_token_count, result.total_output_token_count) for result in results] == [
        (5, 1),
        (25, 5),
    ]


def test_post_edit_batch_passage_halved_recursively() -> None:
    client = MockLlmCompletionClient(
        [
            MalformedCompletionError([], "not json"),
            _create_passage_answer("uno", "dos"),
            _create_passage_answer("tres", "cuatro"),
        ]
    )
    sources = ["one", "two", "three", "four"]

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment(source, "draft") for source in sources], segments_per_request=4
    )

    assert [result.translation for result in results] == ["uno", "dos", "tres", "cuatro"]
    assert _get_prompt(client, 1).endswith("Source (English):\n1. one\n2. two")
    assert _get_prompt(client, 2).endswith("Source (English):\n1. three\n2. four")
    assert '<verse offset="0" role="to_translate" number="1">' in _get_prompt(client, 2)


def test_post_edit_batch_passage_skips_empty_source() -> None:
    client = MockLlmCompletionClient([_create_passage_answer("el perro", "la casa")])
    segments = [
        PostEditSegment("the dog", "la perro"),
        PostEditSegment(" ", " el gato\n negro "),
        PostEditSegment("the house", "el casa"),
    ]

    results = _create_post_editor(client).post_edit_batch(segments, segments_per_request=3)

    assert [result.translation for result in results] == ["el perro", "el gato negro", "la casa"]
    assert [result.succeeded for result in results] == [True, False, True]
    assert '<missing_verse offset="1"/>' in _get_prompt(client)
    assert '<verse offset="2" role="to_translate" number="2">' in _get_prompt(client)


def test_post_edit_batch_passage_failure_fails_run() -> None:
    failure = RuntimeError("connection reset")
    client = MockLlmCompletionClient([failure, _create_answer(translation="el gato")])
    segments = [
        PostEditSegment("the dog", "la perro"),
        PostEditSegment("the house", "el casa"),
        PostEditSegment("the cat", "la gato"),
    ]

    with pytest.raises(PostEditBatchError, match="2 of 3") as error:
        _create_post_editor(client).post_edit_batch(segments, segments_per_request=2)

    assert [result.translation for result in error.value.results] == ["la perro", "el casa", "el gato"]
    assert error.value.errors == {0: failure, 1: failure}


def test_post_edit_batch_segments_per_request_zero() -> None:
    with pytest.raises(ValueError, match="segments_per_request"):
        _create_post_editor(MockLlmCompletionClient([])).post_edit_batch([], segments_per_request=0)


def test_post_edit_batch_unrelated_prompt() -> None:
    client = MockLlmCompletionClient([_create_passage_answer("uno", "dos")])
    segments = [
        PostEditSegment("one", "un", (ContextVerse(-1, "zero", "cero"), ContextVerse(1, "after one", "tras uno"))),
        PostEditSegment("forty", "cuarenta", (ContextVerse(-1, "thirty-nine"),)),
    ]

    _create_post_editor(client).post_edit_batch(segments, segments_per_request=2, consecutive=False)

    prompt = _get_prompt(client)
    assert (
        "<passage_context>\n"
        '  <verse offset="-1">\n'
        '    <source lang="English">zero</source>\n'
        '    <target lang="Spanish">cero</target>\n'
        "  </verse>\n"
        '  <verse offset="0" role="to_translate" number="1">\n'
        '    <source lang="English">one</source>\n'
        "  </verse>\n"
        '  <verse offset="1">\n'
        '    <source lang="English">after one</source>\n'
        '    <target lang="Spanish">tras uno</target>\n'
        "  </verse>\n"
        "</passage_context>\n"
        "\n"
        "<passage_context>\n"
        '  <verse offset="-1">\n'
        '    <source lang="English">thirty-nine</source>\n'
        '    <target lang="Spanish" status="not_available"/>\n'
        "  </verse>\n"
        '  <verse offset="0" role="to_translate" number="2">\n'
        '    <source lang="English">forty</source>\n'
        "  </verse>\n"
        "</passage_context>"
    ) in prompt
    assert prompt.endswith("Source (English):\n1. one\n2. forty")
    system_instruction = client.requests[0]["messages"][0]["content"]
    assert "unrelated parts of the text" in system_instruction
    assert "Read the numbered verses together" not in system_instruction


def test_post_edit_batch_unrelated_halved() -> None:
    client = MockLlmCompletionClient(
        [
            MalformedCompletionError([], "not json"),
            _create_passage_answer("uno", "dos"),
            _create_passage_answer("tres", "cuatro"),
        ]
    )
    sources = ["one", "two", "three", "four"]

    results = _create_post_editor(client).post_edit_batch(
        [PostEditSegment(source, "draft") for source in sources], segments_per_request=4, consecutive=False
    )

    assert [result.translation for result in results] == ["uno", "dos", "tres", "cuatro"]
    assert all("unrelated parts of the text" in request["messages"][0]["content"] for request in client.requests)
    assert _get_prompt(client, 2).count("<passage_context>") == 2
