from machine.corpora import UsfmFileTextCorpus
from machine.tokenization import LatinWordTokenizer

from corpora_test_helpers import USFM_STYLESHEET_PATH, USFM_TEST_PROJECT_PATH


def test_texts() -> None:
    tokenizer = LatinWordTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8-sig", USFM_TEST_PROJECT_PATH)

    assert [t.id for t in corpus.texts] == ["MAT", "MRK"]


def test_get_text() -> None:
    tokenizer = LatinWordTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8-sig", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    assert text.id == "MAT"
    assert corpus.get_text("LUK") is None
