import enum
import warnings
from typing import Any, Callable, Sequence, Union, cast

from transformers import Pipeline
from transformers.tokenization_utils_base import TruncationStrategy

# The following classes are a port of the same classes found in transformers v4


class ReturnType(enum.Enum):
    TENSORS = 0
    TEXT = 1


class TranslationPipeline(Pipeline):

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _sanitize_parameters(
        self,
        src_lang=None,
        tgt_lang=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        forward_params = generate_kwargs

        postprocess_params = {}
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None and self.tokenizer is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        if src_lang is not None:
            preprocess_params["src_lang"] = src_lang
        if tgt_lang is not None:
            preprocess_params["tgt_lang"] = tgt_lang
        if src_lang is None and tgt_lang is None:
            # Backward compatibility, direct arguments use is preferred.
            task = generate_kwargs.get("task", self.task)
            items = task.split("_")
            if task and len(items) == 4:
                # translation, XX, to YY
                preprocess_params["src_lang"] = items[1]
                preprocess_params["tgt_lang"] = items[3]
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, *args: Sequence[Union[str, Sequence[str]]], **kwargs: Any) -> list[dict[str, str]]:
        result = super().__call__(*args, **kwargs)
        if not isinstance(result, list) or not args:
            return cast(list[dict[str, str]], result or [])
        if (
            isinstance(args[0], list)
            and all(isinstance(el, str) for el in args[0])
            and all(isinstance(res, (list, tuple)) and len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]  # type: ignore

        return cast(list[dict[str, str]], result)

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        if input_length > 0.9 * max_length:
            warnings.warn(
                f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider "
                "increasing your max_length manually, e.g. translator('...', max_length=400)"
            )
        return True

    def preprocess(
        self,
        *args,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")
        build_inputs = getattr(self.tokenizer, "_build_translation_inputs", None)
        if callable(build_inputs):
            build_inputs_fn = cast(Callable[..., Any], build_inputs)
            return build_inputs_fn(
                *args, return_tensors="pt", truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            return self._parse_and_tokenize(*args, truncation=truncation)

    def _parse_and_tokenize(self, *args, truncation):
        prefix = self.prefix if self.prefix is not None else ""

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("A tokenizer is required for translation")

        if isinstance(args[0], list):
            if tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise TypeError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )

        inputs = tokenizer(*args, padding=padding, truncation=truncation, return_tensors="pt")

        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        return inputs
