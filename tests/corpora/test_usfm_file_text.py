from machine.corpora import UsfmFileTextCorpus
from machine.scripture.verse_ref import VerseRef
from machine.tokenization import NullTokenizer

from .corpora_test_helpers import USFM_STYLESHEET_PATH, USFM_TEST_PROJECT_PATH


def test_get_segments_nonempty_text() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    segments = list(text.get_segments())

    assert len(segments) == 12

    assert segments[0].segment_ref == VerseRef.from_string("MAT 1:1", corpus.versification)
    assert segments[0].segment[0] == "Chapter one, verse one."

    assert segments[1].segment_ref == VerseRef.from_string("MAT 1:2", corpus.versification)
    assert segments[1].segment[0] == "Chapter one, verse two."

    assert segments[4].segment_ref == VerseRef.from_string("MAT 1:5", corpus.versification)
    assert segments[4].segment[0] == "Chapter one, verse five."

    assert segments[5].segment_ref == VerseRef.from_string("MAT 2:1", corpus.versification)
    assert segments[5].segment[0] == "Chapter two, verse one."

    assert segments[6].segment_ref == VerseRef.from_string("MAT 2:2", corpus.versification)
    assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
    assert segments[6].is_in_range

    assert segments[7].segment_ref == VerseRef.from_string("MAT 2:3", corpus.versification)
    assert len(segments[7].segment) == 0
    assert segments[7].is_in_range

    assert segments[8].segment_ref == VerseRef.from_string("MAT 2:4a", corpus.versification)
    assert len(segments[8].segment) == 0
    assert segments[8].is_in_range

    assert segments[9].segment_ref == VerseRef.from_string("MAT 2:4b", corpus.versification)
    assert segments[9].segment[0] == "Chapter two, verse four."

    assert segments[10].segment_ref == VerseRef.from_string("MAT 2:5", corpus.versification)
    assert segments[10].segment[0] == "Chapter two, verse five."

    assert segments[11].segment_ref == VerseRef.from_string("MAT 2:6", corpus.versification)
    assert segments[11].segment[0] == "Chapter two, verse six."


def test_get_segments_sentence_start() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    segments = list(text.get_segments())

    assert len(segments) == 12

    assert segments[3].segment_ref == VerseRef.from_string("MAT 1:4", corpus.versification)
    assert segments[3].segment[0] == "Chapter one, verse four,"
    assert segments[3].is_sentence_start

    assert segments[4].segment_ref == VerseRef.from_string("MAT 1:5", corpus.versification)
    assert segments[4].segment[0] == "Chapter one, verse five."
    assert not segments[4].is_sentence_start


def test_get_segments_empty_text() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MRK")
    assert text is not None
    segments = list(text.get_segments())

    assert len(segments) == 0


def test_get_segments_include_markers() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, USFM_STYLESHEET_PATH, "utf-8", USFM_TEST_PROJECT_PATH, include_markers=True)

    text = corpus.get_text("MAT")
    assert text is not None
    segments = list(text.get_segments())

    assert len(segments) == 12

    assert segments[0].segment_ref == VerseRef.from_string("MAT 1:1", corpus.versification)
    assert segments[0].segment[0] == "Chapter one, verse one.\\f + \\fr 1:1: \\ft This is a footnote.\\f*"

    assert segments[1].segment_ref == VerseRef.from_string("MAT 1:2", corpus.versification)
    assert segments[1].segment[0] == "Chapter one, \\li2 verse two."

    assert segments[4].segment_ref == VerseRef.from_string("MAT 1:5", corpus.versification)
    assert (
        segments[4].segment[0]
        == 'Chapter one, \\li2 verse \\fig Figure 1|src="image1.png" size="col" ref="1:5"\\fig* five.'
    )

    assert segments[5].segment_ref == VerseRef.from_string("MAT 2:1", corpus.versification)
    assert segments[5].segment[0] == "Chapter \\add two\\add*, verse\\f + \\fr 2:1: \\ft This is a footnote.\\f* one."

    assert segments[6].segment_ref == VerseRef.from_string("MAT 2:2", corpus.versification)
    assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
    assert segments[6].is_in_range

    assert segments[7].segment_ref == VerseRef.from_string("MAT 2:3", corpus.versification)
    assert len(segments[7].segment) == 0
    assert segments[7].is_in_range

    assert segments[8].segment_ref == VerseRef.from_string("MAT 2:4a", corpus.versification)
    assert len(segments[8].segment) == 0
    assert segments[8].is_in_range

    assert segments[9].segment_ref == VerseRef.from_string("MAT 2:4b", corpus.versification)
    assert segments[9].segment[0] == "Chapter two, verse four."

    assert segments[10].segment_ref == VerseRef.from_string("MAT 2:5", corpus.versification)
    assert segments[10].segment[0] == "Chapter two, verse five."

    assert segments[11].segment_ref == VerseRef.from_string("MAT 2:6", corpus.versification)
    assert segments[11].segment[0] == 'Chapter two, verse \\w six|strong="12345" \\w*.'
