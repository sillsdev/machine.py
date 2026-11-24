import re
from math import exp
from threading import Lock
from time import sleep

from machine.translation.llm.example import Example
from machine.translation.llm.example_retriever import ExampleRetriever
from machine.translation.llm.llm_completion_client import Completion, LlmCompletionClient, LlmUsageTotals, TokenLogprob
from machine.translation.llm.openrouter_llm_completion_client import ContentBlockError, RefusalError
from machine.translation.llm.remote_llm_engine import RemoteLlmEngine
from machine.translation.translation_sources import TranslationSources


class _Retriever(ExampleRetriever):
    def __init__(self, examples):
        self.queries = []
        super().__init__(examples)

    def _fit(self) -> None:
        pass

    def _rank(self, query: str, k: int, exclude_index=None):
        self.queries.append(query)
        return self._examples[:k]


class _Client(LlmCompletionClient):
    def __init__(self) -> None:
        self.messages = []
        self.logprobs_requested = False

    def complete(self, messages, logprobs=False):
        self.messages = messages
        self.logprobs_requested = logprobs
        return Completion("Hello, world!", 0.01, [TokenLogprob(" Hello", -0.2), TokenLogprob("!", -0.4)])

    def supports_logprobs(self) -> bool:
        return True


class _BatchClient(LlmCompletionClient):
    def __init__(self, replies, delay=0):
        self.replies = iter(replies)
        self.calls = []
        self.delay = delay
        self.lock = Lock()

    def complete(self, messages, logprobs=False):
        if self.delay:
            sleep(self.delay)
        with self.lock:
            self.calls.append(messages)
            reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _EchoBatchClient(LlmCompletionClient):
    def __init__(self):
        self.calls = []
        self.lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def complete(self, messages, logprobs=False):
        content = messages[-1]["content"]
        sources = re.findall(r"^\d+\. source (\d+)$", content, flags=re.MULTILINE)
        with self.lock:
            self.calls.append(messages)
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        sleep(0.02)
        with self.lock:
            self.active_calls -= 1
        if sources:
            reply = "\n".join(f"{i}. target {source}" for i, source in enumerate(sources, 1))
        else:
            source = re.search(r"<source_to_translate>\nsource (\d+)\n</source_to_translate>", content)
            reply = "" if source is None else f"target {source.group(1)}"
        return Completion(reply, 0.1, input_token_count=100, output_token_count=20)


def test_single_translation_builds_prompt_and_result() -> None:
    client = _Client()
    examples = [Example("buenos días", "good morning"), Example("gracias", "thank you")]
    retriever = _Retriever(examples)
    engine = RemoteLlmEngine(client, retriever, "spa", "eng", num_examples=1)

    result = engine.translate("hola mundo")

    assert [message["role"] for message in client.messages] == ["system", "user"]
    assert "<source_to_translate>\nhola mundo\n</source_to_translate>" in client.messages[-1]["content"]
    assert client.logprobs_requested
    assert retriever.queries == ["hola mundo"]
    assert result.translation == "Hello, world!"
    assert result.source_tokens == ["hola", "mundo"]
    assert result.target_tokens == ["Hello", ",", "world", "!"]
    assert result.sources == [TranslationSources.LLM] * 4
    assert result.confidences == [-1.0] * 4
    assert result.sequence_confidence == exp(-0.3)
    assert result.alignment.row_count == 2
    assert result.alignment.column_count == 4


def test_batches_limit_request_size_preserve_order_and_track_usage():
    sources = [f"source {i}" for i in range(31)]
    client = _EchoBatchClient()
    engine = RemoteLlmEngine(client, _Retriever([]), "eng", "spa", num_workers=3)

    results = engine.translate_batch(sources)

    assert [result.translation for result in results] == [f"target {i}" for i in range(31)]
    assert len(client.calls) == 3
    batch_sizes = [
        len(re.findall(r"^\d+\. source", call[-1]["content"], flags=re.MULTILINE)) or 1 for call in client.calls
    ]
    assert sorted(batch_sizes) == [1, 15, 15]
    assert client.max_active_calls > 1
    assert engine.usage_totals.summary.request_count == 3
    assert engine.usage_totals.summary.input_token_count == 300
    assert engine.usage_totals.summary.output_token_count == 60
    assert round(engine.usage_totals.summary.reported_cost, 6) == 0.3
    assert engine.usage_totals.summary.unpriced_request_count == 0


def test_default_single_prompt_matches_expected_messages():
    client = _Client()
    examples = [
        Example("Love one another & <all>.", "Amaos 'unos' a otros."),
        Example("God so loved the world.", "Porque de tal manera amó Dios al mundo."),
        Example("unused", "unused"),
    ]
    engine = RemoteLlmEngine(client, _Retriever(examples), "eng", "spa", num_examples=2)

    engine.translate("God is love.")

    assert client.messages == [
        {
            "role": "system",
            "content": (
                "Translate the final eng Bible verse into spa, following the translation conventions demonstrated "
                "by this project's parallel examples.\nUse the examples as evidence for spa vocabulary, grammar, "
                "inflection, names, spelling and punctuation. Retain shared words and established borrowings when the "
                "examples support them; use natural target constructions instead of mechanically copying source wording."
                "\nPreserve the full meaning of the source: participants, actions, negation, relationships and emphasis. "
                "Adapt an example's wording to the current sentence; similar examples may describe different people or "
                "events. Nearby translations provide context, not additional content to translate.\nReturn only the "
                "spa translation of the final source verse on one line, without a label, verse number, explanation or "
                "alternative translations."
            ),
        },
        {
            "role": "user",
            "content": (
                "<translation_examples>\neng: God so loved the world.\nspa: Porque de tal manera amó Dios al "
                "mundo.\n\neng: Love one another &amp; &lt;all&gt;.\nspa: Amaos 'unos' a otros.\n"
                "</translation_examples>\n\nFollowing the project's examples above, translate only this eng verse "
                "into spa.\n<source_to_translate>\nGod is love.\n</source_to_translate>\nspa:"
            ),
        },
    ]


def test_batch_prompt_formats_retrieved_examples():
    client = _BatchClient([Completion("1. uno\n2. dos")])
    engine = RemoteLlmEngine(
        client,
        _Retriever([Example("a & <b>", "'c'"), Example("other", "otro")]),
        "eng",
        "spa",
        num_examples=1,
    )

    engine.translate_batch(["one & <two>", "three"])

    assert client.calls[0] == [
        {
            "role": "system",
            "content": (
                "Translate the numbered eng Bible passages at the end into spa, following the translation "
                "conventions demonstrated by this project's parallel examples.\n"
                "Use the examples as evidence for spa vocabulary, grammar, inflection, names, spelling and "
                "punctuation. Retain shared words and established borrowings when the examples support them; use "
                "natural target constructions instead of mechanically copying source wording.\n"
                "Preserve the full meaning of the source: participants, actions, negation, relationships and "
                "emphasis. Adapt an example's wording to the current sentence; similar examples may describe "
                "different people or events.\nReturn only the spa translations, one numbered line per source "
                "passage, without labels, explanations or alternative translations."
            ),
        },
        {
            "role": "user",
            "content": (
                "<translation_examples>\neng: a &amp; &lt;b&gt;\nspa: 'c'\n</translation_examples>\n\n"
                "Following the project's examples above, translate each of these 2 consecutive eng passages into "
                "spa. Some may be section headings rather than verses. Read them together, so that participants, "
                "pronouns, and the flow of the passage stay consistent across them, but translate each one on its own.\n"
                "Reply with exactly 2 lines, one per passage, in the same order, each formatted as "
                "`<number>. <translation>`. Do not merge, split, reorder, or omit passages, and do not add any other "
                "text.\n<source_to_translate>\n1. one &amp; &lt;two&gt;\n2. three\n</source_to_translate>"
            ),
        },
    ]


def test_zero_shot_prompt_has_no_examples_block():
    client = _Client()
    engine = RemoteLlmEngine(client, _Retriever([]), "eng", "spa", num_examples=0)

    engine.translate("bread & <fish>")

    assert client.messages[-1]["content"] == (
        "Translate only this eng verse into spa.\n"
        "<source_to_translate>\nbread &amp; &lt;fish&gt;\n</source_to_translate>\nspa:"
    )


def test_zero_shot_batch_prompt_has_no_examples_block():
    client = _BatchClient([Completion("1. uno\n2. dos")])
    engine = RemoteLlmEngine(client, _Retriever([]), "eng", "spa", num_examples=0)

    engine.translate_batch(["one", "two"])

    assert client.calls[0][1]["content"] == (
        "Translate each of these 2 consecutive eng passages into spa. Some may be section headings "
        "rather than verses. Read them together, so that participants, pronouns, and the flow of the passage "
        "stay consistent across them, but translate each one on its own.\n"
        "Reply with exactly 2 lines, one per passage, in the same order, each formatted as "
        "`<number>. <translation>`. Do not merge, split, reorder, or omit passages, and do not add any other "
        "text.\n<source_to_translate>\n1. one\n2. two\n</source_to_translate>"
    )


def test_prompt_reverses_retrieved_example_order():
    client = _Client()
    engine = RemoteLlmEngine(
        client,
        _Retriever([Example("most", "first"), Example("less", "second"), Example("other", "third")]),
        "eng",
        "spa",
        num_examples=2,
    )

    engine.translate("hello")

    content = client.messages[1]["content"]
    assert content.index("eng: less") < content.index("eng: most")


def test_whole_corpus_goes_into_system_message_in_corpus_order():
    client = _Client()
    retriever = _Retriever([Example("{first}", "one"), Example("second", "two")])
    engine = RemoteLlmEngine(client, retriever, "eng", "spa", num_examples=2)

    engine.translate("hello")

    assert client.messages[0]["content"].endswith(
        "\n\n<translation_examples>\neng: {first}\nspa: one\n\n" "eng: second\nspa: two\n</translation_examples>"
    )
    assert client.messages[1]["content"] == (
        "Translate only this eng verse into spa.\n<source_to_translate>\nhello\n</source_to_translate>\nspa:"
    )
    assert retriever.queries == []


def test_custom_prompt_and_example_format_are_used_for_single_and_batch():
    client = _BatchClient([Completion("uno"), Completion("1. uno\n2. dos")])
    engine = RemoteLlmEngine(
        client,
        _Retriever([Example("a & b", "c"), Example("other", "otro")]),
        "eng",
        "spa",
        num_examples=1,
        prompt={
            "system_message": "Translate {src_lang} to {trg_lang}.",
            "instruction_template": "{examples}|{source}",
            "batch_system_message": "Batch {trg_lang}.",
            "batch_instruction_template": "{num_segments}|{examples}|{source}",
            "example_format": {"type": "text", "template": "{source} => {target}", "escape": False},
        },
    )

    engine.translate("x & y")
    engine.translate_batch(["x", "y"])

    assert client.calls[0] == [
        {"role": "system", "content": "Translate eng to spa."},
        {"role": "user", "content": "a & b => c|x & y"},
    ]
    assert client.calls[1] == [
        {"role": "system", "content": "Batch spa."},
        {"role": "user", "content": "2|a & b => c|1. x\n2. y"},
    ]


def test_custom_empty_system_message_omits_system_turn_for_batches():
    client = _BatchClient([Completion("1. uno\n2. dos")])
    engine = RemoteLlmEngine(client, _Retriever([]), "eng", "spa", num_examples=0, prompt={"system_message": ""})

    engine.translate_batch(["one", "two"])

    assert [message["role"] for message in client.calls[0]] == ["user"]


def test_empty_custom_example_template_is_not_replaced_with_the_default():
    client = _Client()
    engine = RemoteLlmEngine(
        client,
        _Retriever([Example("a", "b"), Example("c", "d")]),
        "eng",
        "spa",
        num_examples=1,
        prompt={"example_format": {"type": "text", "template": ""}, "instruction_template": "[{examples}]{source}"},
    )

    engine.translate("hello")

    assert client.messages[-1]["content"] == "[]hello"


def test_json_example_format_and_back_translation_guidance():
    client = _Client()
    engine = RemoteLlmEngine(
        client,
        _Retriever([Example("a", "b"), Example("c", "d")]),
        "eng",
        "spa",
        num_examples=1,
        prompt={"example_format": "json", "back_translation": True},
    )

    engine.translate("x & y")

    assert "A back-translation shows what the eng text ACTUALLY says" in client.messages[0]["content"]
    assert '[\n  {\n    "source": "a",\n    "target": "b"\n  }\n]\n' in client.messages[1]["content"]
    assert "<source_to_translate>\nx & y\n</source_to_translate>" in client.messages[1]["content"]


def test_malformed_batch_is_corrected_then_split():
    client = _BatchClient(
        [
            Completion("not numbered", 0.01),
            Completion("still malformed", 0.02),
            Completion("uno", 0.03),
            Completion("dos", 0.04),
        ]
    )
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", batch_size=2)

    results = engine.translate_batch(["uno", "dos"])

    assert [result.translation for result in results] == ["uno", "dos"]
    assert len(client.calls) == 4
    assert client.calls[1] == client.calls[0] + [
        {"role": "assistant", "content": "not numbered"},
        {
            "role": "user",
            "content": (
                "That reply did not have the required format. Reply again with exactly 2 lines, one per segment, "
                "in the same order, each formatted as `<number>. <translation>`, and nothing else."
            ),
        },
    ]
    assert engine.usage_totals.summary.request_count == 4
    assert engine.usage_totals.summary.reported_cost == 0.1


def test_batch_refusal_retries_segments_and_reports_failures():
    client = _BatchClient(
        [
            ContentBlockError([]),
            ContentBlockError([]),
            ContentBlockError([]),
            Completion("hola"),
            RefusalError([]),
            RefusalError([]),
            RefusalError([]),
            Completion("later"),
        ]
    )
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", batch_size=2, num_workers=1)

    results = engine.translate_batch(["source one", "source two"])

    assert [result.translation for result in results] == ["hola", ""]
    assert len(client.calls) == 7
    assert results[0].error is None
    assert results[1].error == "Completion response was a refusal from the model."
    assert engine.translate("next").error is None
    assert results[1].error == "Completion response was a refusal from the model."


def test_empty_source_is_skipped_and_empty_reply_is_retried():
    client = _BatchClient([Completion("```text\n```", 0.01), Completion("Translation: hola", 0.02)])
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng")

    results = engine.translate_batch(["", "hola"])

    assert [result.translation for result in results] == ["", "hola"]
    assert [result.error for result in results] == [None, None]
    assert len(client.calls) == 2
    assert client.calls[1] == client.calls[0]
    assert engine.usage_totals.summary.request_count == 2
    assert engine.usage_totals.summary.reported_cost == 0.03


def test_segment_errors_stay_with_results_across_batches():
    client = _BatchClient(
        [ContentBlockError([]), ContentBlockError([]), ContentBlockError([]), Completion("translated")]
    )
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", batch_size=1, num_workers=1)

    results = engine.translate_batch(["", "blocked", "successful"])

    assert [result.translation for result in results] == ["", "", "translated"]
    assert [result.error for result in results] == [
        None,
        "Completion response terminated because of content filtering.",
        None,
    ]


def test_translation_drafts_keep_separate_error_states():
    client = _BatchClient([RefusalError([]), RefusalError([]), RefusalError([]), Completion("translated")])
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng", num_workers=1)

    results = engine.translate_n(2, "source")

    assert [result.translation for result in results] == ["", "translated"]
    assert [result.error for result in results] == ["Completion response was a refusal from the model.", None]


def test_usage_totals_count_reported_tokens_and_unpriced_requests():
    totals = LlmUsageTotals()
    totals.add(Completion("hola", cost=None, input_token_count=None, output_token_count=3))

    summary = totals.summary

    assert summary.request_count == 1
    assert summary.input_token_count == 0
    assert summary.output_token_count == 3
    assert summary.reported_cost == 0
    assert summary.unpriced_request_count == 1


def test_translate_n_batch_treats_a_string_as_one_segment():
    client = _Client()
    engine = RemoteLlmEngine(client, _Retriever([]), "spa", "eng")

    results = engine.translate_n_batch(1, "hola mundo")

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].source_tokens == ["hola", "mundo"]
