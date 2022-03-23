from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmFileTextCorpus


def test_texts() -> None:
    corpus = UsfmFileTextCorpus("usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    assert [t.id for t in corpus.texts] == ["MAT", "MRK"]


def test_get_text() -> None:
    corpus = UsfmFileTextCorpus("usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    mat = corpus.get_text("MAT")
    assert mat is not None
    assert any(mat.get_rows())

    luk = corpus.get_text("LUK")
    assert luk is None
