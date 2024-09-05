from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmTokenizer, UsfmTokenType


def test_tokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    assert len(tokens) == 234

    assert tokens[0].type is UsfmTokenType.BOOK
    assert tokens[0].marker == "id"
    assert tokens[0].data == "MAT"
    assert tokens[0].line_number == 1
    assert tokens[0].column_number == 1

    assert tokens[37].type is UsfmTokenType.TEXT
    assert tokens[37].text == "Chapter One "
    assert tokens[37].line_number == 10
    assert tokens[37].column_number == 4

    assert tokens[38].type is UsfmTokenType.VERSE
    assert tokens[38].marker == "v"
    assert tokens[38].data == "1"
    assert tokens[38].line_number == 11
    assert tokens[38].column_number == 1

    assert tokens[47].type is UsfmTokenType.NOTE
    assert tokens[47].marker == "f"
    assert tokens[47].data == "+"
    assert tokens[47].line_number == 11
    assert tokens[47].column_number == 52


def test_detokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    result = usfm_tokenizer.detokenize(tokens)
    assert result == usfm


def _read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
