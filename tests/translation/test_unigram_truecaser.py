from machine.translation import UnigramTruecaser

TRAINING_SEGMENTS = [
    ["The", "house", "is", "made", "of", "wood", "."],
    ["I", "go", "on", "adventures", "."],
    ["He", "read", "the", "book", "about", "Sherlock", "Holmes", "."],
    ["John", "and", "I", "agree", "that", "you", "and", "I", "are", "smart", "."],
]


def test_truecase_empty() -> None:
    truecaser = _create_truecaser()
    result = truecaser.truecase([])
    assert result == []


def test_truecase_capitialized_name() -> None:
    truecaser = _create_truecaser()
    result = truecaser.truecase(["THE", "ADVENTURES", "OF", "SHERLOCK", "HOLMES"])
    assert result == ["the", "adventures", "of", "Sherlock", "Holmes"]


def test_truecase_unknown_word() -> None:
    truecaser = _create_truecaser()
    result = truecaser.truecase(["THE", "EXPLOITS", "OF", "SHERLOCK", "HOLMES"])
    assert result == ["the", "EXPLOITS", "of", "Sherlock", "Holmes"]


def test_truecase_multiple_sentences() -> None:
    truecaser = _create_truecaser()
    result = truecaser.truecase(["SHERLOCK", "HOLMES", "IS", "SMART", ".", "YOU", "AGREE", "."])
    assert result == ["Sherlock", "Holmes", "is", "smart", ".", "you", "agree", "."]


def test_truecase_ignore_first_word_during_training() -> None:
    truecaser = _create_truecaser()
    result = truecaser.truecase(["HE", "IS", "SMART", "."])
    assert result == ["HE", "is", "smart", "."]


def _create_truecaser() -> UnigramTruecaser:
    truecaser = UnigramTruecaser()
    for segment in TRAINING_SEGMENTS:
        truecaser.train_segment(segment)
    return truecaser
