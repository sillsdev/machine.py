from machine.tokenization import LineSegmentTokenizer


def test_tokenize_empty() -> None:
    tokenizer = LineSegmentTokenizer()
    assert not any(tokenizer.tokenize(""))


def test_tokenize_single_line() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is a test.")) == ["This is a test."]


def test_tokenize_multiple_lines() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence.\nThis is the second sentence.")) == [
        "This is the first sentence.",
        "This is the second sentence.",
    ]


def test_tokenize_ends_with_newline() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is a test.\n")) == ["This is a test."]


def test_tokenize_ends_with_newline_and_space() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is a test.\n ")) == ["This is a test.", " "]


def test_tokenize_ends_with_text_and_space() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence.\nThis is a partial sentence ")) == [
        "This is the first sentence.",
        "This is a partial sentence ",
    ]


def test_tokenize_empty_line() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence.\n\nThis is the third sentence.")) == [
        "This is the first sentence.",
        "",
        "This is the third sentence.",
    ]


def test_tokenize_line_ends_with_space() -> None:
    tokenizer = LineSegmentTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence. \nThis is the second sentence.")) == [
        "This is the first sentence. ",
        "This is the second sentence.",
    ]
