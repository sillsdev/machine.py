from testutils.corpora_test_helpers import TEXT_TEST_PROJECT_PATH

from machine.corpora import MultiKeyRef, TextFileTextCorpus, TextRowFlags


def test_get_rows_nonempty_text_refs() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH)

    text = corpus.get_text("Test1")
    assert text is not None
    rows = list(text.get_rows())

    assert len(rows) == 5

    assert rows[0].ref == MultiKeyRef("Test1", ["s", 1, 1])
    assert rows[0].text == "Section one, sentence one."
    assert rows[0].flags == TextRowFlags.SENTENCE_START

    assert rows[1].ref == MultiKeyRef("Test1", ["s", 1, 2])
    assert rows[1].text == "Section one, sentence two. sentence three."
    assert rows[1].flags == TextRowFlags.SENTENCE_START | TextRowFlags.IN_RANGE | TextRowFlags.RANGE_START

    assert rows[2].ref == MultiKeyRef("Test1", ["s", 1, 3])
    assert rows[2].text == ""
    assert rows[2].flags == TextRowFlags.IN_RANGE

    assert rows[3].ref == MultiKeyRef("Test1", ["s", 2, 1])
    assert rows[3].text == "Section two, sentence one."
    assert rows[3].flags == TextRowFlags.SENTENCE_START

    assert rows[4].ref == MultiKeyRef("Test1", ["s", 2, 2])
    assert rows[4].text == "Section two, sentence two."
    assert rows[4].flags == TextRowFlags.NONE


def test_get_rows_nonempty_text_no_refs() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH)

    text = corpus.get_text("Test3")
    assert text is not None
    rows = list(text.get_rows())

    assert len(rows) == 3

    assert rows[0].ref == MultiKeyRef("Test3", [1])
    assert rows[0].text == "Line one."

    assert rows[1].ref == MultiKeyRef("Test3", [2])
    assert rows[1].text == "Line two."

    assert rows[2].ref == MultiKeyRef("Test3", [3])
    assert rows[2].text == "Line three."


def test_get_rows_empty_text() -> None:
    corpus = TextFileTextCorpus(TEXT_TEST_PROJECT_PATH)

    text = corpus.get_text("Test2")
    assert text is not None
    rows = list(text.get_rows())

    assert len(rows) == 0
