from pydantic import BaseModel


class TrainParams(BaseModel):
    do_train: bool | None = None
    optim: str | None = None
    warmup_steps: int | None = None
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    label_smoothing_factor: float | None = None
    group_by_length: bool | None = None
    gradient_checkpointing: bool | None = None
    lr_scheduler_type: str | None = None
    learning_rate: float | None = None
    fp16: bool | None = None
    tf32: bool | None = None
    save_strategy: str | None = None
    max_steps: int | None = None


class GenerateParams(BaseModel):
    device: int | None = None
    num_beams: int | None = 2
    batch_size: int | None = None
    oom_batch_size_backoff_mult: float | None = None


class TokenizerConfig(BaseModel):
    add_unk_src_tokens: bool | None = None
    add_unk_trg_tokens: bool | None = None


class NmtBuildOptions(BaseModel):
    align_pretranslations: bool | None = None
    tags: list[str] | str | None = None
    use_key_terms: bool | None = None
    parent_model_name: str | None = None
    train_params: TrainParams | None = None
    generate_params: GenerateParams | None = None
    tokenizer: TokenizerConfig | None = None
    attn_implementation: str | None = None
