from pydantic import BaseModel, ConfigDict


class ThotMt(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    word_alignment_model_type: str | None = None
    tokenizer: str | None = None


class SmtBuildOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    tags: list[str] | str | None = None
    use_key_terms: bool | None = None
    thot_mt: ThotMt | None = None
