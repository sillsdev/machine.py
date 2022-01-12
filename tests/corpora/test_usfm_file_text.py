from io import StringIO

from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, verse_ref

from machine.corpora import NullScriptureText, UsfmFileTextCorpus
from machine.scripture import ENGLISH_VERSIFICATION, ORIGINAL_VERSIFICATION, VerseRef, Versification
from machine.tokenization import NullTokenizer


def test_get_segments_nonempty_text() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    segments = list(text.get_segments())

    assert len(segments) == 14

    assert verse_ref(segments[0]).exact_equals(VerseRef.from_string("MAT 1:1", corpus.versification))
    assert segments[0].segment[0] == "Chapter one, verse one."

    assert verse_ref(segments[1]).exact_equals(VerseRef.from_string("MAT 1:2", corpus.versification))
    assert segments[1].segment[0] == "Chapter one, verse two."

    assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert segments[4].segment[0] == "Chapter one, verse five."

    assert verse_ref(segments[5]).exact_equals(VerseRef.from_string("MAT 2:1", corpus.versification))
    assert segments[5].segment[0] == "Chapter two, verse one."

    assert verse_ref(segments[6]).exact_equals(VerseRef.from_string("MAT 2:2", corpus.versification))
    assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
    assert segments[6].is_in_range

    assert verse_ref(segments[7]).exact_equals(VerseRef.from_string("MAT 2:3", corpus.versification))
    assert len(segments[7].segment) == 0
    assert segments[7].is_in_range

    assert verse_ref(segments[8]).exact_equals(VerseRef.from_string("MAT 2:4a", corpus.versification))
    assert len(segments[8].segment) == 0
    assert segments[8].is_in_range

    assert verse_ref(segments[9]).exact_equals(VerseRef.from_string("MAT 2:4b", corpus.versification))
    assert segments[9].segment[0] == "Chapter two, verse four."

    assert verse_ref(segments[10]).exact_equals(VerseRef.from_string("MAT 2:5", corpus.versification))
    assert segments[10].segment[0] == "Chapter two, verse five."

    assert verse_ref(segments[11]).exact_equals(VerseRef.from_string("MAT 2:6", corpus.versification))
    assert segments[11].segment[0] == "Chapter two, verse six."


def test_get_segments_sentence_start() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    segments = list(text.get_segments())

    assert len(segments) == 14

    assert verse_ref(segments[3]).exact_equals(VerseRef.from_string("MAT 1:4", corpus.versification))
    assert segments[3].segment[0] == "Chapter one, verse four,"
    assert segments[3].is_sentence_start

    assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert segments[4].segment[0] == "Chapter one, verse five."
    assert not segments[4].is_sentence_start


def test_get_segments_empty_text() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MRK")
    segments = list(text.get_segments())

    assert len(segments) == 0


def test_get_segments_include_markers() -> None:
    tokenizer = NullTokenizer()
    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH, include_markers=True)

    text = corpus.get_text("MAT")
    segments = list(text.get_segments())

    assert len(segments) == 14

    assert verse_ref(segments[0]).exact_equals(VerseRef.from_string("MAT 1:1", corpus.versification))
    assert (
        segments[0].segment[0]
        == "Chapter \\pn one\\+pro WON\\+pro*\\pn*, verse one.\\f + \\fr 1:1: \\ft This is a footnote.\\f*"
    )

    assert verse_ref(segments[1]).exact_equals(VerseRef.from_string("MAT 1:2", corpus.versification))
    assert segments[1].segment[0] == "Chapter one, \\li2 verse\\f + \\fr 1:2: \\ft This is a footnote.\\f* two."

    assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", corpus.versification))
    assert (
        segments[4].segment[0]
        == 'Chapter one, \\li2 verse \\fig Figure 1|src="image1.png" size="col" ref="1:5"\\fig* five.'
    )

    assert verse_ref(segments[5]).exact_equals(VerseRef.from_string("MAT 2:1", corpus.versification))
    assert segments[5].segment[0] == "Chapter \\add two\\add*, verse \\f + \\fr 2:1: \\ft This is a footnote.\\f*one."

    assert verse_ref(segments[6]).exact_equals(VerseRef.from_string("MAT 2:2", corpus.versification))
    assert segments[6].segment[0] == "Chapter two, verse \\fm ∆\\fm*two. Chapter two, verse three."
    assert segments[6].is_in_range

    assert verse_ref(segments[7]).exact_equals(VerseRef.from_string("MAT 2:3", corpus.versification))
    assert len(segments[7].segment) == 0
    assert segments[7].is_in_range

    assert verse_ref(segments[8]).exact_equals(VerseRef.from_string("MAT 2:4a", corpus.versification))
    assert len(segments[8].segment) == 0
    assert segments[8].is_in_range

    assert verse_ref(segments[9]).exact_equals(VerseRef.from_string("MAT 2:4b", corpus.versification))
    assert segments[9].segment[0] == "Chapter two, verse four."

    assert verse_ref(segments[10]).exact_equals(VerseRef.from_string("MAT 2:5", corpus.versification))
    assert segments[10].segment[0] == "Chapter two, verse five \\rq (MAT 3:1)\\rq*."

    assert verse_ref(segments[11]).exact_equals(VerseRef.from_string("MAT 2:6", corpus.versification))
    assert segments[11].segment[0] == 'Chapter two, verse \\w six|strong="12345" \\w*.'


def test_get_segments_sort_based_on() -> None:
    tokenizer = NullTokenizer()

    src = "MAT 1:2 = MAT 1:3\nMAT 1:3 = MAT 1:2\n"
    stream = StringIO(src)
    versification = Versification("custom", "vers.txt", ENGLISH_VERSIFICATION)
    versification = Versification.parse(stream, "vers.txt", versification, "custom")

    corpus = UsfmFileTextCorpus(tokenizer, "usfm.sty", "utf-8-sig", USFM_TEST_PROJECT_PATH, versification)

    orig_vers_text = NullScriptureText(tokenizer, "MAT", ORIGINAL_VERSIFICATION)

    text = corpus.get_text("MAT")
    segments = list(text.get_segments(sort_based_on=orig_vers_text))

    assert len(segments) == 14

    assert verse_ref(segments[0]).exact_equals(VerseRef.from_string("MAT 1:1", versification))
    assert segments[0].segment[0] == "Chapter one, verse one."

    assert verse_ref(segments[1]).exact_equals(VerseRef.from_string("MAT 1:3", versification))
    assert segments[1].segment[0] == "Chapter one, verse three."

    assert verse_ref(segments[2]).exact_equals(VerseRef.from_string("MAT 1:2", versification))
    assert segments[2].segment[0] == "Chapter one, verse two."

    assert verse_ref(segments[4]).exact_equals(VerseRef.from_string("MAT 1:5", versification))
    assert segments[4].segment[0] == "Chapter one, verse five."

    assert verse_ref(segments[5]).exact_equals(VerseRef.from_string("MAT 2:1", versification))
    assert segments[5].segment[0] == "Chapter two, verse one."

    assert verse_ref(segments[6]).exact_equals(VerseRef.from_string("MAT 2:2", versification))
    assert segments[6].segment[0] == "Chapter two, verse two. Chapter two, verse three."
    assert segments[6].is_in_range

    assert verse_ref(segments[7]).exact_equals(VerseRef.from_string("MAT 2:3", versification))
    assert len(segments[7].segment) == 0
    assert segments[7].is_in_range

    assert verse_ref(segments[8]).exact_equals(VerseRef.from_string("MAT 2:4a", versification))
    assert len(segments[8].segment) == 0
    assert segments[8].is_in_range

    assert verse_ref(segments[9]).exact_equals(VerseRef.from_string("MAT 2:4b", versification))
    assert segments[9].segment[0] == "Chapter two, verse four."

    assert verse_ref(segments[10]).exact_equals(VerseRef.from_string("MAT 2:5", versification))
    assert segments[10].segment[0] == "Chapter two, verse five."

    assert verse_ref(segments[11]).exact_equals(VerseRef.from_string("MAT 2:6", versification))
    assert segments[11].segment[0] == "Chapter two, verse six."
