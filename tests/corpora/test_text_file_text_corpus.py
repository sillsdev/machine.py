import pytest
from testutils.corpora_test_helpers import TEXT_TEST_PATH

from machine.corpora import TextFileTextCorpus


def test_does_not_exist() -> None:
    with pytest.raises(FileNotFoundError):
        TextFileTextCorpus("does-not-exist.txt")


def test_folder() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PATH)
    assert [t.id for t in corpus.texts] == ["text1", "text2"]


def test_single_file() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PATH / "text1.txt")
    assert [t.id for t in corpus.texts] == ["*all*"]


def test_pattern() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PATH / "text?.txt")
    assert [t.id for t in corpus.texts] == ["1", "2"]

    corpus = TextFileTextCorpus(TEXT_TEST_PATH / "*.txt")
    assert [t.id for t in corpus.texts] == ["text1", "text2"]
