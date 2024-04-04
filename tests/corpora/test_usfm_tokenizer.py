from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmTokenizer, UsfmTokenType


def test_tokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    assert len(tokens) == 151

    assert tokens[0].type is UsfmTokenType.BOOK
    assert tokens[0].marker == "id"
    assert tokens[0].data == "MAT"

    assert tokens[10].type is UsfmTokenType.TEXT
    assert tokens[10].text == "Chapter One "

    assert tokens[11].type is UsfmTokenType.VERSE
    assert tokens[11].marker == "v"
    assert tokens[11].data == "1"

    assert tokens[20].type is UsfmTokenType.NOTE
    assert tokens[20].marker == "f"
    assert tokens[20].data == "+"


def test_detokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    result = usfm_tokenizer.detokenize(tokens)
    assert result == usfm


def _read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
