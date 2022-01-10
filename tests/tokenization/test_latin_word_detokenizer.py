from machine.tokenization import LatinWordDetokenizer


def test_detokenize_empty() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize([]) == ""


def test_detokenize_punctuation_at_end_of_word() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(["This", "is", "a", "test", ",", "also", "."]) == "This is a test, also."


def test_detokenize_punctuation_at_start_of_word() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(["Is", "this", "a", "test", "?", "(", "yes", ")"]) == "Is this a test? (yes)"


def test_detokenize_currency_symbol() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(["He", "had", "$", "50", "."]) == "He had $50."


def test_detokenize_quotes() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(['"', "This", "is", "a", "test", ".", '"']) == '"This is a test."'


def test_detokenize_multiple_quotes() -> None:
    detokenizer = LatinWordDetokenizer()
    assert (
        detokenizer.detokenize(["“", "‘", "Moses'", "’", "cat", "said", "‘", "Meow", "’", "to", "the", "dog", ".", "”"])
        == "“‘Moses'’ cat said ‘Meow’ to the dog.”"
    )


def test_detokenize_slash() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(["This", "is", "a", "test", "/", "trial", "."]) == "This is a test/trial."


def test_detokenize_angle_bracket() -> None:
    detokenizer = LatinWordDetokenizer()
    assert detokenizer.detokenize(["This", "is", "a", "<<", "test", ">>", "."]) == "This is a <<test>>."
