from testutils.corpora_test_helpers import verse_ref
from testutils.dbl_bundle_test_environment import DblBundleTestEnvironment

from machine.scripture import VerseRef


def test_get_segments_nonempty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        rows = list(text.get_rows())

        assert len(rows) == 14

        assert verse_ref(rows[0]).exact_equals(VerseRef.from_string("MAT 1:1", env.corpus.versification))
        assert rows[0].text == "Chapter one, verse one."

        assert verse_ref(rows[1]).exact_equals(VerseRef.from_string("MAT 1:2", env.corpus.versification))
        assert rows[1].text == "Chapter one, verse two."

        assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", env.corpus.versification))
        assert rows[4].text == "Chapter one, verse five."

        assert verse_ref(rows[5]).exact_equals(VerseRef.from_string("MAT 2:1", env.corpus.versification))
        assert rows[5].text == "Chapter two, verse one."

        assert verse_ref(rows[6]).exact_equals(VerseRef.from_string("MAT 2:2", env.corpus.versification))
        assert rows[6].text == "Chapter two, verse two. Chapter two, verse three."
        assert rows[6].is_in_range

        assert verse_ref(rows[7]).exact_equals(VerseRef.from_string("MAT 2:3", env.corpus.versification))
        assert len(rows[7].text) == 0
        assert rows[7].is_in_range

        assert verse_ref(rows[8]).exact_equals(VerseRef.from_string("MAT 2:4a", env.corpus.versification))
        assert len(rows[8].text) == 0
        assert rows[8].is_in_range

        assert verse_ref(rows[9]).exact_equals(VerseRef.from_string("MAT 2:4b", env.corpus.versification))
        assert rows[9].text == "Chapter two, verse four."

        assert verse_ref(rows[10]).exact_equals(VerseRef.from_string("MAT 2:5", env.corpus.versification))
        assert rows[10].text == "Chapter two, verse five."

        assert verse_ref(rows[11]).exact_equals(VerseRef.from_string("MAT 2:6", env.corpus.versification))
        assert rows[11].text == "Chapter two, verse six."


def test_get_segments_sentence_start() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        rows = list(text.get_rows())

        assert len(rows) == 14

        assert verse_ref(rows[3]).exact_equals(VerseRef.from_string("MAT 1:4", env.corpus.versification))
        assert rows[3].text == "Chapter one, verse four,"
        assert rows[3].is_sentence_start

        assert verse_ref(rows[4]).exact_equals(VerseRef.from_string("MAT 1:5", env.corpus.versification))
        assert rows[4].text == "Chapter one, verse five."
        assert not rows[4].is_sentence_start


def test_get_segments_empty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MRK")
        assert text is not None
        rows = list(text.get_rows())

        assert len(rows) == 0
