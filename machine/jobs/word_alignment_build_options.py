from pydantic import BaseModel


class ThotAlign(BaseModel):
    word_alignment_heuristic: str | None = None
    model_type: str | None = None
    tokenizer: str | None = None


class WordAlignmentBuildOptions(BaseModel):
    tags: list[str] | str | None = None
    use_key_terms: bool | None = None
    thot_align: ThotAlign | None = None
