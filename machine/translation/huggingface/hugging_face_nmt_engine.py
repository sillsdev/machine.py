from __future__ import annotations

import gc
from math import exp, prod
from typing import Iterable, List, Sequence, Tuple, Union, cast

import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, TranslationPipeline
from transformers.generation import BeamSearchEncoderDecoderOutput, GreedySearchEncoderDecoderOutput
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
        model: Union[PreTrainedModel, StrPath, str],
        **pipeline_kwargs,
    ) -> None:
        if isinstance(model, PreTrainedModel):
            model.eval()
        else:
            model_config = AutoConfig.from_pretrained(str(model), label2id={}, id2label={}, num_labels=0)
            model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(str(model), config=model_config))
        self._tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, use_fast=True)
        if "prefix" not in pipeline_kwargs:
            if (
                "src_lang" in pipeline_kwargs
                and "tgt_lang" in pipeline_kwargs
                and (model.name_or_path.startswith("t5-") or model.name_or_path.startswith("google/mt5-"))
            ):
                src_lang = pipeline_kwargs["src_lang"]
                tgt_lang = pipeline_kwargs["tgt_lang"]
                pipeline_kwargs["prefix"] = f"translate {src_lang} to {tgt_lang}: "
        self._pipeline = _TranslationPipeline(
            model=model,
            tokenizer=self._tokenizer,
            **pipeline_kwargs,
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
            Iterable[Union[List[dict], dict]],
            self._pipeline(segments, num_return_sequences=n),
        ):
            if isinstance(outputs, dict):
                outputs = [outputs]
            segment_results: List[TranslationResult] = []
            for output in outputs:
                input_tokens: Sequence[str] = output["input_tokens"]
                output_length = len(output["translation_tokens"])
                builder = TranslationResultBuilder(input_tokens)
                for token, score in zip(output["translation_tokens"], output["token_scores"]):
                    builder.append_token(token, TranslationSources.NMT, exp(score))
                src_indices = torch.argmax(output["token_attentions"], dim=1).tolist()
                wa_matrix = WordAlignmentMatrix.from_word_pairs(
                    len(input_tokens), output_length, set(zip(src_indices, range(output_length)))
                )
                builder.mark_phrase(Range.create(0, len(input_tokens)), wa_matrix)
                segment_results.append(builder.to_result(output["translation_text"]))
            all_results.append(segment_results)
            i += 1
        return all_results

    def __enter__(self) -> HuggingFaceNmtEngine:
        return self

    def close(self) -> None:
        del self._pipeline
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


class _TranslationPipeline(TranslationPipeline):
    def preprocess(self, *args, truncation=TruncationStrategy.LONGEST_FIRST, src_lang=None, tgt_lang=None):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")
        sentences = [
            s
            if isinstance(s, str)
            else self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(s), use_source_tokenizer=True)
            for s in args
        ]
        inputs = cast(
            BatchEncoding, super().preprocess(*sentences, truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang)
        )
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
        output = self.model.generate(
            **model_inputs,
            **generate_kwargs,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        if isinstance(output, BeamSearchEncoderDecoderOutput):
            output_ids = output.sequences
            beam_indices = output.beam_indices
            scores = output.scores
            attentions = output.cross_attentions
        elif isinstance(output, GreedySearchEncoderDecoderOutput):
            output_ids = output.sequences
            beam_indices = torch.zeros_like(output_ids)
            assert output.scores is not None
            scores = tuple(torch.nn.functional.log_softmax(logits, dim=-1) for logits in output.scores)
            attentions = output.cross_attentions
        else:
            raise RuntimeError("Cannot postprocess the output of the model.")

        assert beam_indices is not None and scores is not None
        out_b = output_ids.shape[0]
        num_beams = scores[0].shape[0] // in_b
        n_sequences = out_b // in_b
        start_index = 0
        if self.model.config.decoder_start_token_id is not None:
            start_index = 1
        indices = torch.stack(
            (
                torch.arange(output_ids.shape[1] - start_index, device=output_ids.device).expand(in_b, n_sequences, -1),
                torch.reshape(beam_indices[:, start_index:] % num_beams, (in_b, n_sequences, -1)),
                torch.reshape(output_ids[:, start_index:], (in_b, n_sequences, -1)),
            ),
            dim=3,
        )
        scores = torch.stack(scores, dim=0).reshape(len(scores), in_b, num_beams, -1).transpose(0, 1)
        scores = torch_gather_nd(scores, indices, 1)
        if self.model.config.decoder_start_token_id is not None:
            scores = torch.cat((torch.zeros(scores.shape[0], scores.shape[1], 1, device=scores.device), scores), dim=2)

        assert attentions is not None
        num_heads = attentions[0][0].shape[1]
        indices = torch.stack(
            (
                torch.arange(output_ids.shape[1] - start_index, device=output_ids.device).expand(in_b, n_sequences, -1),
                torch.reshape(beam_indices[:, start_index:] % num_beams, (in_b, n_sequences, -1)),
            ),
            dim=3,
        )
        num_layers = len(attentions[0])
        layer = (2 * num_layers) // 3
        attentions = (
            torch.stack([cast(Tuple[torch.FloatTensor, ...], a)[layer][:, :, -1, :] for a in attentions], dim=0)
            .squeeze()
            .reshape(len(attentions), in_b, num_beams, num_heads, -1)
            .transpose(0, 1)
        )
        attentions = torch.mean(attentions, dim=3)
        attentions = torch_gather_nd(attentions, indices, 1)
        if self.model.config.decoder_start_token_id is not None:
            attentions = torch.cat(
                (
                    torch.zeros(
                        (attentions.shape[0], attentions.shape[1], 1, attentions.shape[3]), device=attentions.device
                    ),
                    attentions,
                ),
                dim=2,
            )

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
        all_special_ids = set(self.tokenizer.all_special_ids)

        input_ids = model_outputs["input_ids"][0]
        input_indices: List[int] = []
        for i, input_id in enumerate(input_ids):
            id = cast(int, input_id.item())
            if id not in all_special_ids:
                input_indices.append(i)
        input_tokens = model_outputs["input_tokens"][0]

        records = []
        output_ids: torch.Tensor
        scores: torch.Tensor
        attentions: torch.Tensor
        for output_ids, scores, attentions in zip(
            model_outputs["output_ids"][0],
            model_outputs["scores"][0],
            model_outputs["attentions"][0],
        ):
            output_tokens: List[str] = []
            output_indices: List[int] = []
            for i, output_id in enumerate(output_ids):
                id = cast(int, output_id.item())
                if id not in all_special_ids:
                    output_tokens.append(self.tokenizer.convert_ids_to_tokens(id))
                    output_indices.append(i)
            scores = scores[output_indices]
            attentions = attentions[output_indices]
            attentions = attentions[:, input_indices]
            records.append(
                {
                    "input_tokens": input_tokens,
                    "translation_tokens": output_tokens,
                    "token_scores": scores,
                    "token_attentions": attentions,
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
