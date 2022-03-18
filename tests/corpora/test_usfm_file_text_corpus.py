from io import StringIO

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, verse_ref

from machine.corpora import ScriptureTextCorpus, UsfmFileTextCorpus
from machine.scripture import ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION, VerseRef, Versification


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


def test_get_rows_based_on() -> None:
    src = "&MAT 1:4-5 = MAT 1:4\n" "MAT 1:2 = MAT 1:3\n" "MAT 1:3 = MAT 1:2\n"
    stream = StringIO(src)
    versification = Versification("custom", "vers.txt", ENGLISH_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")

    corpus = UsfmFileTextCorpus("usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH, versification)

    orig_vers_corpus = ScriptureTextCorpus(ORIGINAL_VERSIFICATION)

    rows = list(corpus.get_rows(orig_vers_corpus))

    assert len(rows) == 14

    assert verse_ref(rows[0]).exact_equals(VerseRef.from_string("MAT 1:1", versification))
    assert rows[0].text == "Chapter one, verse one."

    assert verse_ref(rows[1]).exact_equals(VerseRef.from_string("MAT 1:3", versification))
    assert rows[1].text == "Chapter one, verse three."

    assert verse_ref(rows[2]).exact_equals(VerseRef.from_string("MAT 1:2", versification))
    assert rows[2].text == "Chapter one, verse two."

    assert verse_ref(rows[3]).exact_equals(VerseRef.from_string("MAT 1:4", versification))
    assert rows[3].segment[0] == "Chapter one, verse four,"
    assert rows[3].segment[1] == "Chapter one, verse five."
    assert rows[3].is_in_range
    assert rows[3].is_range_start

    assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", versification))
    assert len(rows[4].segment) == 0
    assert rows[4].is_in_range
    assert not rows[4].is_range_start

    assert verse_ref(rows[5]).exact_equals(VerseRef.from_string("MAT 2:1", versification))
    assert rows[5].text == "Chapter two, verse one."

    assert verse_ref(rows[6]).exact_equals(VerseRef.from_string("MAT 2:2", versification))
    assert rows[6].text == "Chapter two, verse two. Chapter two, verse three."
    assert rows[6].is_in_range
    assert rows[6].is_range_start

    assert verse_ref(rows[7]).exact_equals(VerseRef.from_string("MAT 2:3", versification))
    assert len(rows[7].segment) == 0
    assert rows[7].is_in_range
    assert not rows[7].is_range_start

    assert verse_ref(rows[8]).exact_equals(VerseRef.from_string("MAT 2:4a", versification))
    assert len(rows[8].segment) == 0
    assert rows[8].is_in_range
    assert not rows[8].is_range_start

    assert verse_ref(rows[9]).exact_equals(VerseRef.from_string("MAT 2:4b", versification))
    assert rows[9].text == "Chapter two, verse four."

    assert verse_ref(rows[10]).exact_equals(VerseRef.from_string("MAT 2:5", versification))
    assert rows[10].text == "Chapter two, verse five."

    assert verse_ref(rows[11]).exact_equals(VerseRef.from_string("MAT 2:6", versification))
    assert rows[11].text == "Chapter two, verse six."
