from testutils.corpora_test_helpers import USFM_TEST_PROJECT_PATH, scripture_ref

from machine.corpora import ScriptureRef, UsfmFileTextCorpus


def test_get_rows_nonempty_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 23

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1", corpus.versification)
    assert rows[0].text == "Chapter one, verse one."

    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:2", corpus.versification)
    assert rows[1].text == "Chapter one, verse two."

    assert scripture_ref(rows[4]) == ScriptureRef.parse("MAT 1:5", corpus.versification)
    assert rows[4].text == "Chapter one, verse five."

    assert scripture_ref(rows[8]) == ScriptureRef.parse("MAT 2:1", corpus.versification)
    assert rows[8].text == "Chapter two, verse one."

    assert scripture_ref(rows[9]) == ScriptureRef.parse("MAT 2:2", corpus.versification)
    assert rows[9].text == "Chapter two, verse two. Chapter two, verse three."
    assert rows[9].is_in_range
    assert rows[9].is_range_start

    assert scripture_ref(rows[10]) == ScriptureRef.parse("MAT 2:3", corpus.versification)
    assert len(rows[10].segment) == 0
    assert rows[10].is_in_range
    assert not rows[10].is_range_start

    assert scripture_ref(rows[11]) == ScriptureRef.parse("MAT 2:4a", corpus.versification)
    assert len(rows[11].segment) == 0
    assert rows[11].is_in_range
    assert not rows[11].is_range_start

    assert scripture_ref(rows[12]) == ScriptureRef.parse("MAT 2:4b", corpus.versification)
    assert rows[12].text == "Chapter two, verse four."

    assert scripture_ref(rows[13]) == ScriptureRef.parse("MAT 2:5", corpus.versification)
    assert rows[13].text == "Chapter two, verse five."

    assert scripture_ref(rows[14]) == ScriptureRef.parse("MAT 2:6", corpus.versification)
    assert rows[14].text == "Chapter two, verse six."

    assert scripture_ref(rows[18]) == ScriptureRef.parse("MAT 2:9", corpus.versification)
    assert rows[18].text == "Chapter 2 verse 9"

    assert scripture_ref(rows[19]) == ScriptureRef.parse("MAT 2:10", corpus.versification)
    assert rows[19].text == "Chapter 2 verse 10"

    assert scripture_ref(rows[20]) == ScriptureRef.parse("MAT 2:11", corpus.versification)
    assert not rows[20].text


def test_get_rows_nonempty_text_all_text() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH, include_all_text=True)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 49

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:0/1:h", corpus.versification)
    assert rows[0].text == "Matthew"

    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:0/2:mt", corpus.versification)
    assert rows[1].text == "Matthew"

    assert scripture_ref(rows[2]) == ScriptureRef.parse("MAT 1:0/3:ip", corpus.versification)
    assert rows[2].text == "An introduction to Matthew"

    assert scripture_ref(rows[3]) == ScriptureRef.parse("MAT 1:0/3:ip/1:fe", corpus.versification)
    assert rows[3].text == "This is an endnote."

    assert scripture_ref(rows[4]) == ScriptureRef.parse("Mat 1:0/4:p", corpus.versification)
    assert rows[4].text == "MAT 1 Here is another paragraph."

    assert scripture_ref(rows[7]) == ScriptureRef.parse("MAT 1:0/7:weirdtaglookingthing", corpus.versification)
    assert rows[7].text == "that is not an actual tag."

    assert scripture_ref(rows[8]) == ScriptureRef.parse("MAT 1:0/8:s", corpus.versification)
    assert rows[8].text == "Chapter One"

    assert scripture_ref(rows[10]) == ScriptureRef.parse("MAT 1:1/1:f", corpus.versification)
    assert rows[10].text == "1:1: This is a footnote."

    assert scripture_ref(rows[12]) == ScriptureRef.parse("MAT 1:2/1:f", corpus.versification)
    assert rows[12].text == "1:2: This is a footnote."

    assert scripture_ref(rows[19]) == ScriptureRef.parse("MAT 2:0/1:tr/1:tc1", corpus.versification)
    assert rows[19].text == "Row one, column one."

    assert scripture_ref(rows[20]) == ScriptureRef.parse("MAT 2:0/1:tr/2:tc2", corpus.versification)
    assert rows[20].text == "Row one, column two."

    assert scripture_ref(rows[21]) == ScriptureRef.parse("MAT 2:0/2:tr/1:tc1", corpus.versification)
    assert rows[21].text == "Row two, column one."

    assert scripture_ref(rows[22]) == ScriptureRef.parse("MAT 2:0/2:tr/2:tc2", corpus.versification)
    assert rows[22].text == "Row two, column two."

    assert scripture_ref(rows[23]) == ScriptureRef.parse("MAT 2:0/3:s1", corpus.versification)
    assert rows[23].text == "Chapter Two"

    assert scripture_ref(rows[24]) == ScriptureRef.parse("MAT 2:0/4:p", corpus.versification)
    assert not rows[24].text

    assert scripture_ref(rows[26]) == ScriptureRef.parse("MAT 2:1/1:f", corpus.versification)
    assert rows[26].text == "2:1: This is a footnote."

    assert scripture_ref(rows[29]) == ScriptureRef.parse("MAT 2:3/1:esb/1:ms", corpus.versification)
    assert rows[29].text == "This is a sidebar"

    assert scripture_ref(rows[30]) == ScriptureRef.parse("MAT 2:3/1:esb/2:p", corpus.versification)
    assert rows[30].text == "Here is some sidebar content."

    assert scripture_ref(rows[36]) == ScriptureRef.parse("MAT 2:7a/1:s", corpus.versification)
    assert rows[36].text == "Section header"

    assert scripture_ref(rows[43]) == ScriptureRef.parse("MAT 2:12/1:restore", corpus.versification)
    assert rows[43].text == "restore information"


def test_get_rows_sentence_start() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 23

    assert scripture_ref(rows[3]) == ScriptureRef.parse("MAT 1:4", corpus.versification)
    assert rows[3].text == "Chapter one, verse four,"
    assert rows[3].is_sentence_start

    assert scripture_ref(rows[4]) == ScriptureRef.parse("MAT 1:5", corpus.versification)
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

    assert len(rows) == 23

    assert scripture_ref(rows[0]) == ScriptureRef.parse("MAT 1:1", corpus.versification)
    assert (
        rows[0].text == "Chapter \\pn one\\+pro WON\\+pro*\\pn*, verse one.\\f + \\fr 1:1: \\ft This is a footnote.\\f*"
    )

    assert scripture_ref(rows[1]) == ScriptureRef.parse("MAT 1:2", corpus.versification)
    assert rows[1].text == "\\bd C\\bd*hapter one, \\li2 verse\\f + \\fr 1:2: \\ft This is a footnote.\\f* two."

    assert scripture_ref(rows[4]) == ScriptureRef.parse("MAT 1:5", corpus.versification)
    assert rows[4].text == 'Chapter one, \\li2 verse \\fig Figure 1|src="image1.png" size="col" ref="1:5"\\fig* five.'

    assert scripture_ref(rows[8]) == ScriptureRef.parse("MAT 2:1", corpus.versification)
    assert rows[8].text == "Chapter \\add two\\add*, verse \\f + \\fr 2:1: \\ft This is a footnote.\\f*one."

    assert scripture_ref(rows[9]) == ScriptureRef.parse("MAT 2:2", corpus.versification)
    assert rows[9].text == "Chapter two, // verse \\fm ∆\\fm*two. Chapter two, verse \\w three|lemma\\w*."
    assert rows[9].is_in_range
    assert rows[9].is_range_start

    assert scripture_ref(rows[10]) == ScriptureRef.parse("MAT 2:3", corpus.versification)
    assert len(rows[10].segment) == 0
    assert rows[10].is_in_range
    assert not rows[10].is_range_start

    assert scripture_ref(rows[11]) == ScriptureRef.parse("MAT 2:4a", corpus.versification)
    assert len(rows[11].segment) == 0
    assert rows[11].is_in_range
    assert not rows[11].is_range_start

    assert scripture_ref(rows[12]) == ScriptureRef.parse("MAT 2:4b", corpus.versification)
    assert rows[12].text == "Chapter two, verse four."

    assert scripture_ref(rows[13]) == ScriptureRef.parse("MAT 2:5", corpus.versification)
    assert rows[13].text == "Chapter two, verse five \\rq (MAT 3:1)\\rq*."

    assert scripture_ref(rows[14]) == ScriptureRef.parse("MAT 2:6", corpus.versification)
    assert rows[14].text == 'Chapter two, verse \\w six|strong="12345" \\w*.'

    assert scripture_ref(rows[18]) == ScriptureRef.parse("MAT 2:9", corpus.versification)
    assert rows[18].text == "Chapter\\tcr2 2\\tc3 verse\\tcr4 9"

    assert scripture_ref(rows[19]) == ScriptureRef.parse("MAT 2:10", corpus.versification)
    assert rows[19].text == "\\tc3-4 Chapter 2 verse 10"


def test_get_rows_include_markers_all_text() -> None:

    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH, include_markers=True, include_all_text=True)

    text = corpus.get_text("MAT")
    assert text is not None
    rows = list(text)

    assert len(rows) == 45

    assert scripture_ref(rows[2]) == ScriptureRef.parse("MAT 1:0/3:ip", corpus.versification)
    assert rows[2].text == "An introduction to Matthew\\fe + \\ft This is an endnote.\\fe*"

    assert scripture_ref(rows[8]) == ScriptureRef.parse("MAT 1:1", corpus.versification)
    assert (
        rows[8].text == "Chapter \\pn one\\+pro WON\\+pro*\\pn*, verse one.\\f + \\fr 1:1: \\ft This is a footnote.\\f*"
    )

    assert scripture_ref(rows[9]) == ScriptureRef.parse("MAT 1:2", corpus.versification)
    assert rows[9].text == "\\bd C\\bd*hapter one, \\li2 verse\\f + \\fr 1:2: \\ft This is a footnote.\\f* two."

    assert scripture_ref(rows[12]) == ScriptureRef.parse("MAT 1:5", corpus.versification)
    assert rows[12].text == 'Chapter one, \\li2 verse \\fig Figure 1|src="image1.png" size="col" ref="1:5"\\fig* five.'

    assert scripture_ref(rows[20]) == ScriptureRef.parse("MAT 2:0/3:s1", corpus.versification)
    assert rows[20].text == "Chapter \\it Two \\it*"

    assert scripture_ref(rows[22]) == ScriptureRef.parse("MAT 2:1", corpus.versification)
    assert rows[22].text == "Chapter \\add two\\add*, verse \\f + \\fr 2:1: \\ft This is a footnote.\\f*one."

    assert scripture_ref(rows[26]) == ScriptureRef.parse("MAT 2:3/1:esb/2:p", corpus.versification)
    assert rows[26].text == "Here is some sidebar // content."


def test_usfm_file_text_corpus_lowercase_usfm_id() -> None:
    corpus = UsfmFileTextCorpus(USFM_TEST_PROJECT_PATH)

    text = corpus.get_text("LEV")
    assert text is not None
    rows = list(text)

    assert len(rows) == 2

    assert scripture_ref(rows[0]) == ScriptureRef.parse("LEV 14:55", corpus.versification)
    assert rows[0].text == "Chapter fourteen, verse fifty-five. Segment b."

    assert scripture_ref(rows[1]) == ScriptureRef.parse("LEV 14:56", corpus.versification)
    assert rows[1].text == "Chapter fourteen, verse fifty-six."
