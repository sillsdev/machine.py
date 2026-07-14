from pydantic import BaseModel, ConfigDict


class ThotAlign(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    word_alignment_heuristic: str | None = None
    model_type: str | None = None
    tokenizer: str | None = None


class WordAlignmentBuildOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    tags: list[str] | str | None = None
    use_key_terms: bool | None = None
    thot_align: ThotAlign | None = None
