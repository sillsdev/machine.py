from pydantic import BaseModel


class ThotMt(BaseModel):
    word_alignment_model_type: str | None = None
    tokenizer: str | None = None


class SmtBuildOptions(BaseModel):
    tags: list[str] | str | None = None
    use_key_terms: bool | None = None
    thot_mt: ThotMt | None = None
