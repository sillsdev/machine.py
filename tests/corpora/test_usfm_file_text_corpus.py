from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmFileTextCorpus
from machine.tokenization import LatinWordTokenizer


def test_texts() -> None:
    tokenizer = LatinWordTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    assert [t.id for t in corpus.texts] == ["MAT", "MRK"]


def test_get_text() -> None:
    tokenizer = LatinWordTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    assert any(corpus.get_text("MAT").get_segments())
    assert not any(corpus.get_text("LUK").get_segments())
