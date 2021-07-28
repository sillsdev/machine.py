from machine.tokenization import LatinSentenceTokenizer


def test_tokenize_empty() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert not any(tokenizer.tokenize(""))


def test_tokenize_single_line() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize("This is a test.")) == ["This is a test."]


def test_tokenize_multiple_lines() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence.\nThis is the second sentence.")) == [
        "This is the first sentence.",
        "This is the second sentence.",
    ]


def test_tokenize_two_sentences() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence. This is the second sentence.")) == [
        "This is the first sentence.",
        "This is the second sentence.",
    ]


def test_tokenize_quotes() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize('"This is the first sentence." This is the second sentence.')) == [
        '"This is the first sentence."',
        "This is the second sentence.",
    ]


def test_tokenize_quotation_in_sentence() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize('"This is the first sentence!" he said. This is the second sentence.')) == [
        '"This is the first sentence!" he said.',
        "This is the second sentence.",
    ]


def test_tokenize_parens() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize("This is the first sentence. (This is the second sentence.)")) == [
        "This is the first sentence.",
        "(This is the second sentence.)",
    ]


def test_tokenize_abbreviation() -> None:
    tokenizer = LatinSentenceTokenizer(abbreviations={"mr", "dr", "ms"})
    assert list(tokenizer.tokenize("Mr. Smith went to Washington. This is the second sentence.")) == [
        "Mr. Smith went to Washington.",
        "This is the second sentence.",
    ]


def test_tokenize_incomplete_sentence() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize("This is an incomplete sentence ")) == ["This is an incomplete sentence "]


def test_tokenize_complete_sentence_with_space_at_end() -> None:
    tokenizer = LatinSentenceTokenizer()
    assert list(tokenizer.tokenize('"This is a complete sentence." \n')) == ['"This is a complete sentence."']
