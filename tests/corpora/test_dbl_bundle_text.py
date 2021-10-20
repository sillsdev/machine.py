from machine.scripture import VerseRef
from tests.corpora.dbl_bundle_test_environment import DblBundleTestEnvironment


def test_get_segments_nonempty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 14

        assert segments[0].segment_ref == VerseRef.from_string("MAT 1:1", env.corpus.versification)
        assert segments[0].segment[0] == "Chapter one, verse one."

        assert segments[1].segment_ref == VerseRef.from_string("MAT 1:2", env.corpus.versification)
        assert segments[1].segment[0] == "Chapter one, verse two."

        assert segments[4].segment_ref == VerseRef.from_string("MAT 1:5", env.corpus.versification)
        assert segments[4].segment[0] == "Chapter one, verse five."

        assert segments[5].segment_ref == VerseRef.from_string("MAT 2:1", env.corpus.versification)
        assert segments[5].segment[0] == "Chapter two, verse one."

        assert segments[6].segment_ref == VerseRef.from_string("MAT 2:2", env.corpus.versification)
        assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
        assert segments[6].is_in_range

        assert segments[7].segment_ref == VerseRef.from_string("MAT 2:3", env.corpus.versification)
        assert len(segments[7].segment) == 0
        assert segments[7].is_in_range

        assert segments[8].segment_ref == VerseRef.from_string("MAT 2:4a", env.corpus.versification)
        assert len(segments[8].segment) == 0
        assert segments[8].is_in_range

        assert segments[9].segment_ref == VerseRef.from_string("MAT 2:4b", env.corpus.versification)
        assert segments[9].segment[0] == "Chapter two, verse four."

        assert segments[10].segment_ref == VerseRef.from_string("MAT 2:5", env.corpus.versification)
        assert segments[10].segment[0] == "Chapter two, verse five."

        assert segments[11].segment_ref == VerseRef.from_string("MAT 2:6", env.corpus.versification)
        assert segments[11].segment[0] == "Chapter two, verse six."


def test_get_segments_sentence_start() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 14

        assert segments[3].segment_ref == VerseRef.from_string("MAT 1:4", env.corpus.versification)
        assert segments[3].segment[0] == "Chapter one, verse four,"
        assert segments[3].is_sentence_start

        assert segments[4].segment_ref == VerseRef.from_string("MAT 1:5", env.corpus.versification)
        assert segments[4].segment[0] == "Chapter one, verse five."
        assert not segments[4].is_sentence_start


def test_get_segments_empty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MRK")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 0
