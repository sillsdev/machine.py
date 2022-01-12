from testutils.corpora_test_helpers import verse_ref
from testutils.dbl_bundle_test_environment import DblBundleTestEnvironment

from machine.scripture import VerseRef


def test_get_segments_nonempty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        segments = list(text.get_segments())

        assert len(segments) == 14

        assert verse_ref(segments[0]).exact_equals(VerseRef.from_string("MAT 1:1", env.corpus.versification))
        assert segments[0].segment[0] == "Chapter one, verse one."

        assert verse_ref(segments[1]).exact_equals(VerseRef.from_string("MAT 1:2", env.corpus.versification))
        assert segments[1].segment[0] == "Chapter one, verse two."

        assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", env.corpus.versification))
        assert segments[4].segment[0] == "Chapter one, verse five."

        assert verse_ref(segments[5]).exact_equals(VerseRef.from_string("MAT 2:1", env.corpus.versification))
        assert segments[5].segment[0] == "Chapter two, verse one."

        assert verse_ref(segments[6]).exact_equals(VerseRef.from_string("MAT 2:2", env.corpus.versification))
        assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
        assert segments[6].is_in_range

        assert verse_ref(segments[7]).exact_equals(VerseRef.from_string("MAT 2:3", env.corpus.versification))
        assert len(segments[7].segment) == 0
        assert segments[7].is_in_range

        assert verse_ref(segments[8]).exact_equals(VerseRef.from_string("MAT 2:4a", env.corpus.versification))
        assert len(segments[8].segment) == 0
        assert segments[8].is_in_range

        assert verse_ref(segments[9]).exact_equals(VerseRef.from_string("MAT 2:4b", env.corpus.versification))
        assert segments[9].segment[0] == "Chapter two, verse four."

        assert verse_ref(segments[10]).exact_equals(VerseRef.from_string("MAT 2:5", env.corpus.versification))
        assert segments[10].segment[0] == "Chapter two, verse five."

        assert verse_ref(segments[11]).exact_equals(VerseRef.from_string("MAT 2:6", env.corpus.versification))
        assert segments[11].segment[0] == "Chapter two, verse six."


def test_get_segments_sentence_start() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        segments = list(text.get_segments())

        assert len(segments) == 14

        assert verse_ref(segments[3]).exact_equals(VerseRef.from_string("MAT 1:4", env.corpus.versification))
        assert segments[3].segment[0] == "Chapter one, verse four,"
        assert segments[3].is_sentence_start

        assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", env.corpus.versification))
        assert segments[4].segment[0] == "Chapter one, verse five."
        assert not segments[4].is_sentence_start


def test_get_segments_empty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MRK")
        segments = list(text.get_segments())

        assert len(segments) == 0
