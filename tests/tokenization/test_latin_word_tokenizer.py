from machine.tokenization import LatinWordTokenizer


def test_tokenize_empty() -> None:
    tokenizer = LatinWordTokenizer()
    assert not any(tokenizer.tokenize(""))


def test_tokenize_whitespace() -> None:
    tokenizer = LatinWordTokenizer()
    assert not any(tokenizer.tokenize(" "))


def test_tokenize_punctuation_at_end_of_word() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This is a test, also.")) == ["This", "is", "a", "test", ",", "also", "."]


def test_tokenize_punctuation_at_start_of_word() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("Is this a test? (yes)")) == [
        "Is",
        "this",
        "a",
        "test",
        "?",
        "(",
        "yes",
        ")",
    ]


def test_tokenize_punctuation_inside_word() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This isn't a test.")) == ["This", "isn't", "a", "test", "."]
    assert list(tokenizer.tokenize("He had $5,000.")) == ["He", "had", "$", "5,000", "."]


def test_tokenize_symbol() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("He had $50.")) == ["He", "had", "$", "50", "."]


def test_tokenize_abbreviation() -> None:
    tokenizer = LatinWordTokenizer(["mr", "dr", "ms"])
    assert list(tokenizer.tokenize("Mr. Smith went to Washington.")) == [
        "Mr.",
        "Smith",
        "went",
        "to",
        "Washington",
        ".",
    ]


def test_tokenize_quotes() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize('"This is a test."')) == ['"', "This", "is", "a", "test", ".", '"']


def test_tokenize_apostrophe_not_as_single_quote() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("“Moses' cat said ‘Meow’ to the dog.”")) == [
        "“",
        "Moses'",
        "cat",
        "said",
        "‘",
        "Meow",
        "’",
        "to",
        "the",
        "dog",
        ".",
        "”",
    ]
    assert list(tokenizer.tokenize("i ha''on 'ot ano'.")) == ["i", "ha''on", "'ot", "ano'", "."]


def test_tokenize_apostrophe_as_single_quote() -> None:
    tokenizer = LatinWordTokenizer(treat_apostrophe_as_single_quote=True)
    assert list(tokenizer.tokenize("'Moses's cat said 'Meow' to the dog.'")) == [
        "'",
        "Moses's",
        "cat",
        "said",
        "'",
        "Meow",
        "'",
        "to",
        "the",
        "dog",
        ".",
        "'",
    ]


def test_tokenize_slash() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This is a test/trial.")) == ["This", "is", "a", "test", "/", "trial", "."]


def test_tokenize_angle_bracket() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This is a <<test>>.")) == ["This", "is", "a", "<<", "test", ">>", "."]


def test_tokenize_email_address() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This is an email address, name@test.com, in a sentence.")) == [
        "This",
        "is",
        "an",
        "email",
        "address",
        ",",
        "name@test.com",
        ",",
        "in",
        "a",
        "sentence",
        ".",
    ]


def test_tokenize_email_address_at_end_of_sentence() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("Here is an email address: name@test.com.")) == [
        "Here",
        "is",
        "an",
        "email",
        "address",
        ":",
        "name@test.com",
        ".",
    ]


def test_tokenize_url() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("This is a url, http://www.test.com/page.html, in a sentence.")) == [
        "This",
        "is",
        "a",
        "url",
        ",",
        "http://www.test.com/page.html",
        ",",
        "in",
        "a",
        "sentence",
        ".",
    ]


def test_tokenize_url_at_end_of_sentence() -> None:
    tokenizer = LatinWordTokenizer()
    assert list(tokenizer.tokenize("Here is a url: http://www.test.com/page.html?param=1.")) == [
        "Here",
        "is",
        "a",
        "url",
        ":",
        "http://www.test.com/page.html?param=1",
        ".",
    ]
