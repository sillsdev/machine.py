from typing import cast

from transformers import M2M100Config, M2M100ForConditionalGeneration, NllbTokenizer, set_seed


def create_tiny_nllb_model(model_dir: str) -> None:
    # hf-internal-testing/tiny-random-nllb declares MBartTokenizer, which in transformers 5 discards the normalizer in
    # tokenizer.json. Pair its tiny config with the real NLLB tokenizer so that input normalization is exercised.
    tokenizer = cast(NllbTokenizer, NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"))
    config = cast(M2M100Config, M2M100Config.from_pretrained("hf-internal-testing/tiny-random-nllb"))
    config.vocab_size = len(tokenizer)
    config.bos_token_id = cast(int, tokenizer.bos_token_id)
    config.pad_token_id = cast(int, tokenizer.pad_token_id)
    config.eos_token_id = cast(int, tokenizer.eos_token_id)
    config.decoder_start_token_id = config.eos_token_id
    set_seed(0)
    model = M2M100ForConditionalGeneration(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
