from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH

from machine.corpora import ParatextTextCorpus, extract_scripture_corpus
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef


def test_extract_scripture_corpus() -> None:
    corpus = ParatextTextCorpus(USFM_TEST_PROJECT_PATH, include_all_text=True)

    lines = list(extract_scripture_corpus(corpus))
    assert len(lines) == 41899

    text, orig_vref, corpus_vref = lines[0]
    assert text == ""
    assert orig_vref == VerseRef.from_string("GEN 1:1", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("GEN 1:1", corpus.versification)

    text, orig_vref, corpus_vref = lines[3167]
    assert text == "Chapter fourteen, verse fifty-five. Segment b."
    assert orig_vref == VerseRef.from_string("LEV 14:56", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("LEV 14:55", corpus.versification)

    text, orig_vref, corpus_vref = lines[10726]
    assert text == "Chapter twelve, verses three through seven."
    assert orig_vref == VerseRef.from_string("1CH 12:3", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("1CH 12:3", corpus.versification)

    text, orig_vref, corpus_vref = lines[10727]
    assert text == "<range>"
    assert orig_vref == VerseRef.from_string("1CH 12:4", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("1CH 12:4", corpus.versification)

    text, orig_vref, corpus_vref = lines[10731]
    assert text == "<range>"
    assert orig_vref == VerseRef.from_string("1CH 12:8", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("1CH 12:7", corpus.versification)

    text, orig_vref, corpus_vref = lines[10732]
    assert text == "Chapter twelve, verse eight."
    assert orig_vref == VerseRef.from_string("1CH 12:9", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("1CH 12:8", corpus.versification)

    text, orig_vref, corpus_vref = lines[23213]
    assert text == "Chapter one, verse one."
    assert orig_vref == VerseRef.from_string("MAT 1:1", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("MAT 1:1", corpus.versification)

    text, orig_vref, corpus_vref = lines[23240]
    assert text == "<range>"
    assert orig_vref == VerseRef.from_string("MAT 2:3", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("MAT 2:3", corpus.versification)

    text, orig_vref, corpus_vref = lines[23248]
    assert text == ""
    assert orig_vref == VerseRef.from_string("MAT 2:11", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("MAT 2:11", corpus.versification)

    text, orig_vref, corpus_vref = lines[23249]
    assert text == ""
    assert orig_vref == VerseRef.from_string("MAT 2:12", ORIGINAL_VERSIFICATION)
    assert corpus_vref is not None and corpus_vref == VerseRef.from_string("MAT 2:12", corpus.versification)
