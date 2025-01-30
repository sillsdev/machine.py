from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmTokenizer, UsfmTokenType


def test_tokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    assert len(tokens) == 236

    assert tokens[0].type is UsfmTokenType.BOOK
    assert tokens[0].marker == "id"
    assert tokens[0].data == "MAT"
    assert tokens[0].line_number == 1
    assert tokens[0].column_number == 1

    assert tokens[30].type is UsfmTokenType.TEXT
    assert tokens[30].text == "Chapter One "
    assert tokens[30].line_number == 9
    assert tokens[30].column_number == 4

    assert tokens[31].type is UsfmTokenType.VERSE
    assert tokens[31].marker == "v"
    assert tokens[31].data == "1"
    assert tokens[31].line_number == 10
    assert tokens[31].column_number == 1

    assert tokens[40].type is UsfmTokenType.NOTE
    assert tokens[40].marker == "f"
    assert tokens[40].data == "+"
    assert tokens[40].line_number == 10
    assert tokens[40].column_number == 48


def test_detokenize() -> None:
    usfm = _read_usfm()
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    result = usfm_tokenizer.detokenize(tokens)
    assert result == usfm


def test_tokenize_ending_paragraph_marker() -> None:
    usfm = r"""\id MAT - Test
\c 1
\v 1 Descriptive title\x - \xo 18:16 \xt  hello world\x*\p
"""
    tokens = UsfmTokenizer().tokenize(usfm)
    assert len(tokens) == 13


def _read_usfm() -> str:
    with (USFM_TEST_PROJECT_PATH / "41MATTes.SFM").open("r", encoding="utf-8-sig", newline="\r\n") as file:
        return file.read()
