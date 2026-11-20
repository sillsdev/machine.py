from typing import Dict, Sequence

from .post_edit_result import PostEditResult


class PostEditBatchError(Exception):
    def __init__(self, results: Sequence[PostEditResult], errors: Dict[int, Exception]) -> None:
        super().__init__(f"{len(errors)} of {len(results)} segments failed to post-edit.")
        self.results = results
        self.errors = errors
