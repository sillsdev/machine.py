from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, verse_ref

from machine.corpora import UsfmFileTextCorpus
from machine.scripture import VerseRef


def test_get_rows_nonempty_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(corpus)

    assert len(rows) == 17

    assert verse_ref(rows[0]).exact_equals(VerseRef.from_string("MAT 1:1", corpus.versification))
    assert rows[0].text == "Chapter one, verse one."

    assert verse_ref(rows[1]).exact_equals(VerseRef.from_string("MAT 1:2", corpus.versification))
    assert rows[1].text == "Chapter one, verse two."

    assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert rows[4].text == "Chapter one, verse five."

    assert verse_ref(rows[5]).exact_equals(VerseRef.from_string("MAT 2:1", corpus.versification))
    assert rows[5].text == "Chapter two, verse one."

    assert verse_ref(rows[6]).exact_equals(VerseRef.from_string("MAT 2:2", corpus.versification))
    assert rows[6].text == "Chapter two, verse two. Chapter two, verse three."
    assert rows[6].is_in_range
    assert rows[6].is_range_start

    assert verse_ref(rows[7]).exact_equals(VerseRef.from_string("MAT 2:3", corpus.versification))
    assert len(rows[7].segment) == 0
    assert rows[7].is_in_range
    assert not rows[7].is_range_start

    assert verse_ref(rows[8]).exact_equals(VerseRef.from_string("MAT 2:4a", corpus.versification))
    assert len(rows[8].segment) == 0
    assert rows[8].is_in_range
    assert not rows[8].is_range_start

    assert verse_ref(rows[9]).exact_equals(VerseRef.from_string("MAT 2:4b", corpus.versification))
    assert rows[9].text == "Chapter two, verse four."

    assert verse_ref(rows[10]).exact_equals(VerseRef.from_string("MAT 2:5", corpus.versification))
    assert rows[10].text == "Chapter two, verse five."

    assert verse_ref(rows[11]).exact_equals(VerseRef.from_string("MAT 2:6", corpus.versification))
    assert rows[11].text == "Chapter two, verse six."

    assert verse_ref(rows[15]).exact_equals(VerseRef.from_string("MAT 2:9", corpus.versification))
    assert rows[15].text == "Chapter 2 verse 9"

    assert verse_ref(rows[16]).exact_equals(VerseRef.from_string("MAT 2:10", corpus.versification))
    assert rows[16].text == "Chapter 2 verse 10"


def test_get_rows_sentence_start() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 17

    assert verse_ref(rows[3]).exact_equals(VerseRef.from_string("MAT 1:4", corpus.versification))
    assert rows[3].text == "Chapter one, verse four,"
    assert rows[3].is_sentence_start

    assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert rows[4].text == "Chapter one, verse five."
    assert not rows[4].is_sentence_start


def test_get_rows_empty_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MRK")
    assert text is not None
    rows = list(text)

    assert len(rows) == 0


def test_get_rows_include_markers() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH, include_markers=True)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 17

    assert verse_ref(rows[0]).exact_equals(VerseRef.from_string("MAT 1:1", corpus.versification))
    assert (
        rows[0].text == "Chapter \\pn one\\+pro WON\\+pro*\\pn*, verse one.\\f + \\fr 1:1: \\ft This is a footnote.\\f*"
    )

    assert verse_ref(rows[1]).exact_equals(VerseRef.from_string("MAT 1:2", corpus.versification))
    assert rows[1].text == "\\bd C\\bd*hapter one, \\li2 verse\\f + \\fr 1:2: \\ft This is a footnote.\\f* two."

    assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert rows[4].text == 'Chapter one, \\li2 verse \\fig Figure 1|src="image1.png" size="col" ref="1:5"\\fig* five.'

    assert verse_ref(rows[5]).exact_equals(VerseRef.from_string("MAT 2:1", corpus.versification))
    assert rows[5].text == "Chapter \\add two\\add*, verse \\f + \\fr 2:1: \\ft This is a footnote.\\f*one."

    assert verse_ref(rows[6]).exact_equals(VerseRef.from_string("MAT 2:2", corpus.versification))
    assert rows[6].text == "Chapter two, verse \\fm ∆\\fm*two. Chapter two, verse three."
    assert rows[6].is_in_range
    assert rows[6].is_range_start

    assert verse_ref(rows[7]).exact_equals(VerseRef.from_string("MAT 2:3", corpus.versification))
    assert len(rows[7].segment) == 0
    assert rows[7].is_in_range
    assert not rows[7].is_range_start

    assert verse_ref(rows[8]).exact_equals(VerseRef.from_string("MAT 2:4a", corpus.versification))
    assert len(rows[8].segment) == 0
    assert rows[8].is_in_range
    assert not rows[8].is_range_start

    assert verse_ref(rows[9]).exact_equals(VerseRef.from_string("MAT 2:4b", corpus.versification))
    assert rows[9].text == "Chapter two, verse four."

    assert verse_ref(rows[10]).exact_equals(VerseRef.from_string("MAT 2:5", corpus.versification))
    assert rows[10].text == "Chapter two, verse five \\rq (MAT 3:1)\\rq*."

    assert verse_ref(rows[11]).exact_equals(VerseRef.from_string("MAT 2:6", corpus.versification))
    assert rows[11].text == 'Chapter two, verse \\w six|strong="12345" \\w*.'

    assert verse_ref(rows[15]).exact_equals(VerseRef.from_string("MAT 2:9", corpus.versification))
    assert rows[15].text == "Chapter\\tcr2 2\\tc3 verse\\tcr4 9"

    assert verse_ref(rows[16]).exact_equals(VerseRef.from_string("MAT 2:10", corpus.versification))
    assert rows[16].text == "\\tc3-4 Chapter 2 verse 10"
