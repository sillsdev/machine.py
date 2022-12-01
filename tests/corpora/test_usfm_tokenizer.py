from machine.corpora import UsfmTokenizer, UsfmTokenType


def test_tokenize() -> None:
    usfm = """\\id MAT - Test
\\h Matthew
\\mt Matthew
\\ip An introduction to Matthew
\\c 1
\\s Chapter One
\\p
\\v 1 This is verse one of chapter one.
""".replace(
        "\n", "\r\n"
    )

    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    assert len(tokens) == 14

    assert tokens[0].type is UsfmTokenType.BOOK
    assert tokens[0].marker == "id"
    assert tokens[0].data == "MAT"

    assert tokens[10].type is UsfmTokenType.TEXT
    assert tokens[10].text == "Chapter One "

    assert tokens[12].type is UsfmTokenType.VERSE
    assert tokens[12].marker == "v"
    assert tokens[12].data == "1"


def test_detokenize() -> None:
    usfm = """\\id MAT - Test
\\h Matthew
\\mt Matthew
\\ip An introduction to Matthew
\\c 1
\\s Chapter One
\\p
\\v 1 This is verse one of chapter one.
""".replace(
        "\n", "\r\n"
    )
    usfm_tokenizer = UsfmTokenizer()
    tokens = usfm_tokenizer.tokenize(usfm)
    result = usfm_tokenizer.detokenize(tokens)
    assert result == usfm
