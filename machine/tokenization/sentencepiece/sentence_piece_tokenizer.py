from typing import Iterable, Optional

import sentencepiece as sp

from ...annotations.range import Range
from ...utils.typeshed import StrPath
from ..tokenizer import Tokenizer


class SentencePieceTokenizer(Tokenizer[str, int, str]):
    def __init__(self, model_filename: StrPath) -> None:
        self._sp = sp.SentencePieceProcessor()
        self._sp.Load(str(model_filename))

    def tokenize(self, data: str, data_range: Optional[Range[int]] = None) -> Iterable[str]:
        if data_range is None:
            data_range = Range.create(0, len(data))

        data = data[data_range.start : data_range.end]
        return self._sp.EncodeAsPieces(data)
