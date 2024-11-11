from pytest import raises
from testutils.corpora_test_helpers import USFM_INVALID_ID_PROJECT_PATH, USFM_MISMATCH_ID_PROJECT_PATH

from machine.corpora import ParatextTextCorpus


def test_paratext_text_corpus_invalid_id() -> None:
    with raises(ValueError, match=r"The \\id tag in .* is invalid."):
        ParatextTextCorpus(USFM_INVALID_ID_PROJECT_PATH, include_all_text=True)


def test_paratext_text_corpus_mismatch_id() -> None:
    with raises(ValueError, match=r"The \\id tag .* in .* does not match filename book id .*"):
        ParatextTextCorpus(USFM_MISMATCH_ID_PROJECT_PATH, include_all_text=True)
