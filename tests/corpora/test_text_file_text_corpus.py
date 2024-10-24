from pytest import raises
from testutils.corpora_test_helpers import TEXT_TEST_PROJECT_PATH

from machine.corpora import TextFileTextCorpus


def test_does_not_exist() -> None:
    with raises(FileNotFoundError):
        TextFileTextCorpus("does-not-exist.txt")


def test_folder() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH)
    assert [t.id for t in corpus.texts] == ["Test1", "Test2", "Test3"]


def test_single_file() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH / "Test1.txt")
    assert [t.id for t in corpus.texts] == ["*all*"]


def test_pattern_star() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH / "*.txt")
    assert [t.id for t in corpus.texts] == ["Test1", "Test2", "Test3"]


def test_pattern_question_mark() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH / "Test?.txt")
    assert [t.id for t in corpus.texts] == ["1", "2", "3"]
