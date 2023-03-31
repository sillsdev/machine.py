from __future__ import annotations

from math import exp, prod
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline
from transformers.generation import BeamSearchEncoderDecoderOutput
from transformers.tokenization_utils import BatchEncoding, TruncationStrategy

from ...annotations.range import Range
from ...utils.typeshed import StrPath
from ..translation_engine import TranslationEngine
from ..translation_result import TranslationResult
from ..translation_result_builder import TranslationResultBuilder
from ..translation_sources import TranslationSources
from ..word_alignment_matrix import WordAlignmentMatrix


class HuggingFaceNmtEngine(TranslationEngine):
    def __init__(
        self,
        model_name_or_path: StrPath,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=model_config)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self._pipeline = _TranslationPipeline(
            model=model,
            tokenizer=self._tokenizer,
            src_lang=source_lang,
            tgt_lang=target_lang,
            device=device,
            batch_size=batch_size,
        )

    def translate(self, segment: Union[str, Sequence[str]]) -> TranslationResult:
        return self.translate_batch([segment])[0]

    def translate_n(self, n: int, segment: Union[str, Sequence[str]]) -> Sequence[TranslationResult]:
        return self.translate_n_batch(n, [segment])[0]

    def translate_batch(self, segments: Sequence[Union[str, Sequence[str]]]) -> Sequence[TranslationResult]:
        return [results[0] for results in self.translate_n_batch(1, segments)]

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        all_results: List[List[TranslationResult]] = []
        i = 0
        for outputs in cast(
            Iterable[List[dict]],
            self._pipeline(segments, num_return_sequences=n),
        ):
            segment_results: List[TranslationResult] = []
            for output in outputs:
                input_tokens: Sequence[str] = output["input_tokens"]
                output_length = len(output["translation_tokens"])
                builder = TranslationResultBuilder()
                for token, score in zip(output["translation_tokens"], output["token_scores"]):
                    builder.append_token(token, TranslationSources.NMT, exp(score))
                src_indices = torch.argmax(output["token_attentions"], dim=1).tolist()
                wa_matrix = WordAlignmentMatrix.from_word_pairs(
                    len(input_tokens), output_length, set(zip(src_indices, range(output_length)))
                )
                builder.mark_phrase(Range.create(0, len(input_tokens)), wa_matrix)
                segment_results.append(builder.to_result(output["translation_text"], input_tokens))
            all_results.append(segment_results)
            i += 1
        return all_results

    def __enter__(self) -> HuggingFaceNmtEngine:
        return self


class _TranslationPipeline(TranslationPipeline):
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")
        sentences = [
            s
            if isinstance(s, str)
            else self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(s), use_source_tokenizer=True)
            for s in args
        ]
        inputs = cast(BatchEncoding, super().preprocess(*sentences, truncation, src_lang, tgt_lang))
        if inputs.is_fast:
            inputs["input_tokens"] = [
                inputs.tokens(i) if isinstance(args[i], str) else args[i] for i in range(len(args))
            ]
        else:
            inputs["input_tokens"] = [self.tokenizer.tokenize(s) if isinstance(s, str) else s for s in args]
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape

        input_tokens = model_inputs["input_tokens"]
        del model_inputs["input_tokens"]
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", self.model.config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", self.model.config.max_length)
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        output = cast(
            BeamSearchEncoderDecoderOutput,
            self.model.generate(
                **model_inputs,
                **generate_kwargs,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
            ),
        )
        output_ids = output.sequences

        assert output.sequences is not None and output.beam_indices is not None and output.scores is not None
        out_b = output_ids.shape[0]
        num_beams = output.scores[0].shape[0] // in_b
        n_sequences = out_b // in_b
        indices = torch.stack(
            (
                torch.arange(output.sequences.shape[1] - 1, device=output_ids.device).expand(in_b, n_sequences, -1),
                torch.reshape(output.beam_indices[:, :-1] % num_beams, (in_b, n_sequences, -1)),
                torch.reshape(output.sequences[:, 1:], (in_b, n_sequences, -1)),
            ),
            dim=3,
        )
        scores = torch.stack(output.scores, dim=0).reshape(len(output.scores), in_b, num_beams, -1).transpose(0, 1)
        scores = torch_gather_nd(scores, indices, 1)

        assert output.cross_attentions is not None
        num_heads = output.cross_attentions[0][0].shape[1]
        indices = torch.stack(
            (
                torch.arange(1, output.sequences.shape[1], device=output_ids.device).expand(in_b, n_sequences, -1),
                torch.reshape(output.beam_indices[:, :-1] % num_beams, (in_b, n_sequences, -1)),
            ),
            dim=3,
        )
        attentions = (
            torch.stack([cast(Tuple[torch.FloatTensor, ...], a)[4] for a in output.cross_attentions], dim=0)
            .squeeze()
            .reshape(len(output.cross_attentions), in_b, num_beams, num_heads, -1)
            .transpose(0, 1)
        )
        attentions = torch.mean(attentions, dim=3)
        attentions = torch_gather_nd(attentions, indices, 1)

        output_ids = output_ids.reshape(in_b, n_sequences, *output_ids.shape[1:])
        return {
            "input_ids": model_inputs["input_ids"],
            "input_tokens": input_tokens,
            "output_ids": output_ids,
            "scores": scores,
            "attentions": attentions,
        }

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=False):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")

        input_tokens = model_outputs["input_tokens"][0]
        records = []
        for output_ids, scores, attentions in zip(
            model_outputs["output_ids"][0],
            model_outputs["scores"][0],
            model_outputs["attentions"][0],
        ):
            tokens = self.tokenizer.convert_ids_to_tokens(output_ids, skip_special_tokens=True)
            records.append(
                {
                    "input_tokens": input_tokens,
                    "translation_tokens": tokens,
                    "token_scores": scores[: len(tokens)],
                    "token_attentions": attentions[: len(tokens), 1 : len(input_tokens) + 1],
                    "translation_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    ),
                }
            )
        return records


def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """
    torch_gather_nd implements tf.gather_nd in PyTorch.

    This supports multiple batch dimensions as well as multiple channel dimensions.
    """
    index_shape = indices.shape[:-1]
    num_dim = indices.size(-1)
    tail_sizes = params.shape[batch_dim + num_dim :]

    # flatten extra dimensions
    for s in tail_sizes:
        row_indices = torch.arange(s, device=params.device)
        indices = indices.unsqueeze(-2)
        indices = indices.repeat(*[1 for _ in range(indices.dim() - 2)], s, 1)
        row_indices = row_indices.expand(*indices.shape[:-2], -1).unsqueeze(-1)
        indices = torch.cat((indices, row_indices), dim=-1)
        num_dim += 1

    # flatten indices and params to batch specific ones instead of channel specific
    for i in range(num_dim):
        size = prod(params.shape[batch_dim + i + 1 : batch_dim + num_dim])
        indices[..., i] *= size

    indices = indices.sum(dim=-1)
    params = params.flatten(batch_dim, -1)
    indices = indices.flatten(batch_dim, -1)

    out = torch.gather(params, dim=batch_dim, index=indices)
    return out.reshape(*index_shape, *tail_sizes)
