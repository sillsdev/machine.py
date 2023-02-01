from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import UsfmFileTextCorpus, extract_scripture_corpus
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef


def test_texts() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    assert [t.id for t in corpus.texts] == ["1CH", "MAT", "MRK"]


def test_get_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    mat = corpus.get_text("MAT")
    assert mat is not None
    assert any(mat.get_rows())

    luk = corpus.get_text("LUK")
    assert luk is None


def test_extract_scripture_corpus() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    lines = list(extract_scripture_corpus(corpus))
    assert len(lines) == 41899

    text, orig_vref, corpus_vref = lines[0]
    assert text == ""
    assert orig_vref.exact_equals(VerseRef.from_string("GEN 1:1", ORIGINAL_VERSIFICATION))
    assert corpus_vref is None

    text, orig_vref, corpus_vref = lines[10726]
    assert text == "Chapter twelve, verses three through seven."
    assert orig_vref.exact_equals(VerseRef.from_string("1CH 12:3", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("1CH 12:3", corpus.versification))

    text, orig_vref, corpus_vref = lines[10727]
    assert text == "<range>"
    assert orig_vref.exact_equals(VerseRef.from_string("1CH 12:4", ORIGINAL_VERSIFICATION))
    assert corpus_vref is None

    text, orig_vref, corpus_vref = lines[10731]
    assert text == "<range>"
    assert orig_vref.exact_equals(VerseRef.from_string("1CH 12:8", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("1CH 12:7", corpus.versification))

    text, orig_vref, corpus_vref = lines[10732]
    assert text == "Chapter twelve, verse eight."
    assert orig_vref.exact_equals(VerseRef.from_string("1CH 12:9", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("1CH 12:8", corpus.versification))

    text, orig_vref, corpus_vref = lines[23213]
    assert text == "Chapter one, verse one."
    assert orig_vref.exact_equals(VerseRef.from_string("MAT 1:1", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("MAT 1:1", corpus.versification))

    text, orig_vref, corpus_vref = lines[23240]
    assert text == "<range>"
    assert orig_vref.exact_equals(VerseRef.from_string("MAT 2:3", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("MAT 2:3", corpus.versification))

    text, orig_vref, corpus_vref = lines[23248]
    assert text == ""
    assert orig_vref.exact_equals(VerseRef.from_string("MAT 2:11", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("MAT 2:11", corpus.versification))

    text, orig_vref, corpus_vref = lines[23249]
    assert text == ""
    assert orig_vref.exact_equals(VerseRef.from_string("MAT 2:12", ORIGINAL_VERSIFICATION))
    assert corpus_vref is not None and corpus_vref.exact_equals(VerseRef.from_string("MAT 2:12", corpus.versification))
