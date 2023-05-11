from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pytest
import sentencepiece as sp

from machine.tokenization.sentencepiece import SentencePieceTokenizer

TEST_FILENAME = Path(__file__).parent / "data" / "test.txt"


@pytest.fixture(scope="module")
def model_filename() -> Iterable[Path]:
    with TemporaryDirectory() as temp_dir:
        model_filename = Path(temp_dir) / "sp"
        sp.SentencePieceTrainer.Train(f"--input={TEST_FILENAME} --model_prefix={model_filename} --vocab_size=100")
        yield model_filename.with_suffix(".model")


def test_tokenize(model_filename: Path) -> None:
    tokenizer = SentencePieceTokenizer(model_filename)
    tokens = list(tokenizer.tokenize("Other travelling salesmen live a life of luxury."))
    assert len(tokens) == 29


def test_tokenize_empty(model_filename: Path) -> None:
    tokenizer = SentencePieceTokenizer(model_filename)
    tokens = list(tokenizer.tokenize(""))
    assert len(tokens) == 0
