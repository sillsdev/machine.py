from machine.corpora import UsfmFileTextCorpus
from tests.testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH


def test_texts() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    assert [t.id for t in corpus.texts] == ["LEV", "1CH", "MAT", "MRK"]


def test_get_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    mat = corpus.get_text("MAT")
    assert mat is not None
    assert any(mat.get_rows())

    luk = corpus.get_text("LUK")
    assert luk is None
