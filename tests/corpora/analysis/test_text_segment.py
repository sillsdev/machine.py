from machine.corpora import UsfmToken, UsfmTokenType
from machine.corpora.analysis import TextSegment, UsfmMarkerType


def test_builder_initialization() -> None:
    builder = TextSegment.Builder()

    assert builder._text_segment._text == ""
    assert builder._text_segment._previous_segment is None
    assert builder._text_segment._next_segment is None
    assert builder._text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert builder._text_segment._markers_in_preceding_context == set()
    assert builder._text_segment._index_in_verse == 0
    assert builder._text_segment._num_segments_in_verse == 0
    assert builder._text_segment._usfm_token is None


def test_builder_set_text() -> None:
    builder = TextSegment.Builder()
    text = "Example text"
    builder.set_text(text)

    assert builder._text_segment._text == text


def test_builder_set_previous_segment() -> None:
    builder = TextSegment.Builder()
    previous_segment = TextSegment.Builder().set_text("previous segment text").build()
    builder.set_previous_segment(previous_segment)

    assert builder._text_segment._previous_segment == previous_segment
    assert builder._text_segment._next_segment is None
    assert builder._text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert builder._text_segment._markers_in_preceding_context == set()
    assert builder._text_segment._index_in_verse == 0
    assert builder._text_segment._num_segments_in_verse == 0


def test_builder_add_preceding_marker() -> None:
    builder = TextSegment.Builder()
    builder.add_preceding_marker(UsfmMarkerType.CHAPTER)

    assert builder._text_segment._immediate_preceding_marker is UsfmMarkerType.CHAPTER
    assert builder._text_segment._markers_in_preceding_context == {UsfmMarkerType.CHAPTER}
    assert builder._text_segment._previous_segment is None
    assert builder._text_segment._next_segment is None

    builder.add_preceding_marker(UsfmMarkerType.VERSE)
    assert builder._text_segment._immediate_preceding_marker == UsfmMarkerType.VERSE
    assert builder._text_segment._markers_in_preceding_context == {
        UsfmMarkerType.CHAPTER,
        UsfmMarkerType.VERSE,
    }
    assert builder._text_segment._previous_segment is None
    assert builder._text_segment._next_segment is None


def test_builder_set_usfm_token() -> None:
    builder = TextSegment.Builder()
    builder.set_usfm_token(UsfmToken(type=UsfmTokenType.TEXT, text="USFM token text"))

    assert builder._text_segment._usfm_token is not None
    assert builder._text_segment._usfm_token.type == UsfmTokenType.TEXT
    assert builder._text_segment._usfm_token.text == "USFM token text"
    assert builder._text_segment._text == ""
    assert builder._text_segment._previous_segment is None
    assert builder._text_segment._next_segment is None


def test_set_previous_segment() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    previous_segment = TextSegment.Builder().set_text("previous segment text").build()
    text_segment.set_previous_segment(previous_segment)

    assert text_segment._previous_segment == previous_segment
    assert text_segment._next_segment is None
    assert text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert text_segment._markers_in_preceding_context == set()
    assert text_segment._index_in_verse == 0
    assert text_segment._num_segments_in_verse == 0


def test_set_next_segment() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    next_segment = TextSegment.Builder().set_text("next segment text").build()
    text_segment.set_next_segment(next_segment)

    assert text_segment._previous_segment is None
    assert text_segment._next_segment == next_segment
    assert text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert text_segment._markers_in_preceding_context == set()
    assert text_segment._index_in_verse == 0
    assert text_segment._num_segments_in_verse == 0


def test_set_index_in_verse() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    text_segment.set_index_in_verse(2)

    assert text_segment._index_in_verse == 2
    assert text_segment._previous_segment is None
    assert text_segment._next_segment is None
    assert text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert text_segment._markers_in_preceding_context == set()
    assert text_segment._num_segments_in_verse == 0


def test_set_num_segments_in_verse() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    text_segment.set_num_segments_in_verse(5)

    assert text_segment._num_segments_in_verse == 5
    assert text_segment._previous_segment is None
    assert text_segment._next_segment is None
    assert text_segment._immediate_preceding_marker is UsfmMarkerType.NO_MARKER
    assert text_segment._markers_in_preceding_context == set()
    assert text_segment._index_in_verse == 0


def test_equals() -> None:
    basic_segment = TextSegment.Builder().set_text("text1").build()
    same_text_segment = TextSegment.Builder().set_text("text1").build()
    different_text_segment = TextSegment.Builder().set_text("different text").build()

    assert basic_segment == basic_segment
    assert basic_segment != UsfmToken(type=UsfmTokenType.TEXT, text="text1")
    assert basic_segment == same_text_segment
    assert basic_segment != different_text_segment

    segment_with_index = TextSegment.Builder().set_text("text1").build()
    segment_with_index.set_index_in_verse(1)
    segment_with_same_index = TextSegment.Builder().set_text("text1").build()
    segment_with_same_index.set_index_in_verse(1)
    segment_with_different_index = TextSegment.Builder().set_text("text1").build()
    segment_with_different_index.set_index_in_verse(2)

    assert segment_with_index == segment_with_same_index
    assert segment_with_index != segment_with_different_index
    assert segment_with_index != basic_segment

    segment_with_preceding_marker = (
        TextSegment.Builder().set_text("text1").add_preceding_marker(UsfmMarkerType.VERSE).build()
    )
    segment_with_same_preceding_marker = (
        TextSegment.Builder().set_text("text1").add_preceding_marker(UsfmMarkerType.VERSE).build()
    )
    segment_with_different_preceding_marker = (
        TextSegment.Builder().set_text("text1").add_preceding_marker(UsfmMarkerType.CHAPTER).build()
    )
    segment_with_multiple_preceding_markers = (
        TextSegment.Builder()
        .set_text("text1")
        .add_preceding_marker(UsfmMarkerType.CHAPTER)
        .add_preceding_marker(UsfmMarkerType.VERSE)
        .build()
    )

    usfm_token = UsfmToken(type=UsfmTokenType.TEXT, text="USFM token text")
    segment_with_usfm_token = TextSegment.Builder().set_text("text1").set_usfm_token(usfm_token).build()
    segment_with_same_usfm_token = TextSegment.Builder().set_text("text1").set_usfm_token(usfm_token).build()
    segment_with_different_usfm_token = (
        TextSegment.Builder()
        .set_text("text1")
        .set_usfm_token(UsfmToken(type=UsfmTokenType.TEXT, text="Different USFM token text"))
        .build()
    )

    assert segment_with_usfm_token == segment_with_same_usfm_token
    assert segment_with_usfm_token != segment_with_different_usfm_token
    assert basic_segment != segment_with_usfm_token

    # attributes that are not used in equality checks
    segment_with_num_verses = TextSegment.Builder().set_text("text1").build()
    segment_with_num_verses.set_num_segments_in_verse(3)
    segment_with_same_num_verses = TextSegment.Builder().set_text("text1").build()
    segment_with_same_num_verses.set_num_segments_in_verse(3)
    segment_with_different_num_verses = TextSegment.Builder().set_text("text1").build()
    segment_with_different_num_verses.set_num_segments_in_verse(4)

    assert segment_with_num_verses == segment_with_same_num_verses
    assert segment_with_num_verses != segment_with_different_num_verses
    assert segment_with_num_verses != basic_segment

    assert segment_with_preceding_marker == segment_with_same_preceding_marker
    assert segment_with_preceding_marker != segment_with_different_preceding_marker
    assert segment_with_preceding_marker == segment_with_multiple_preceding_markers
    assert segment_with_preceding_marker != basic_segment

    segment_with_previous_segment = TextSegment.Builder().set_text("text1").build()
    segment_with_previous_segment.set_previous_segment(segment_with_num_verses)

    segment_with_next_segment = TextSegment.Builder().set_text("text1").build()
    segment_with_next_segment.set_next_segment(segment_with_num_verses)

    assert basic_segment == segment_with_previous_segment
    assert basic_segment == segment_with_next_segment


def test_get_text() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    assert text_segment.text == "example text"

    text_segment = TextSegment.Builder().set_text("new example text").build()
    assert text_segment.text == "new example text"


def test_length() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    assert text_segment.length == len("example text")

    text_segment = TextSegment.Builder().set_text("new example text").build()
    assert text_segment.length == len("new example text")


def test_substring_before() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    assert text_segment.substring_before(7) == "example"
    assert text_segment.substring_before(8) == "example "
    assert text_segment.substring_before(0) == ""
    assert text_segment.substring_before(12) == "example text"


def test_substring_after() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    assert text_segment.substring_after(7) == " text"
    assert text_segment.substring_after(8) == "text"
    assert text_segment.substring_after(0) == "example text"
    assert text_segment.substring_after(12) == ""
    assert text_segment.substring_after(11) == "t"


def test_is_marker_in_preceding_context() -> None:
    no_preceding_marker_segment = TextSegment.Builder().set_text("example text").build()
    assert no_preceding_marker_segment.marker_is_in_preceding_context(UsfmMarkerType.CHAPTER) is False
    assert no_preceding_marker_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE) is False
    assert no_preceding_marker_segment.marker_is_in_preceding_context(UsfmMarkerType.CHARACTER) is False

    one_preceding_marker_text_segment = (
        TextSegment.Builder().set_text("example text").add_preceding_marker(UsfmMarkerType.CHARACTER).build()
    )

    assert one_preceding_marker_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHARACTER) is True
    assert one_preceding_marker_text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE) is False
    assert one_preceding_marker_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHAPTER) is False

    two_preceding_markers_text_segment = (
        TextSegment.Builder()
        .set_text("example text")
        .add_preceding_marker(UsfmMarkerType.CHAPTER)
        .add_preceding_marker(UsfmMarkerType.VERSE)
        .build()
    )
    assert two_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHAPTER) is True
    assert two_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE) is True
    assert two_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHARACTER) is False

    three_preceding_markers_text_segment = (
        TextSegment.Builder()
        .set_text("example text")
        .add_preceding_marker(UsfmMarkerType.CHAPTER)
        .add_preceding_marker(UsfmMarkerType.VERSE)
        .add_preceding_marker(UsfmMarkerType.CHARACTER)
        .build()
    )
    assert three_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHAPTER) is True
    assert three_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.VERSE) is True
    assert three_preceding_markers_text_segment.marker_is_in_preceding_context(UsfmMarkerType.CHARACTER) is True


def test_is_first_segment_in_verse() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    text_segment.set_index_in_verse(0)
    assert text_segment.is_first_segment_in_verse() is True

    text_segment.set_index_in_verse(1)
    assert text_segment.is_first_segment_in_verse() is False


def test_is_last_segment_in_verse() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    text_segment.set_index_in_verse(0)
    text_segment.set_num_segments_in_verse(1)
    assert text_segment.is_last_segment_in_verse() is True

    text_segment.set_index_in_verse(0)
    text_segment.set_num_segments_in_verse(2)
    assert text_segment.is_last_segment_in_verse() is False

    text_segment.set_index_in_verse(1)
    assert text_segment.is_last_segment_in_verse() is True


def test_replace_substring() -> None:
    text_segment = TextSegment.Builder().set_text("example text").build()
    text_segment.replace_substring(0, 7, "sample")
    assert text_segment.text == "sample text"

    text_segment.replace_substring(7, 11, "text")
    assert text_segment.text == "sample text"

    text_segment.replace_substring(0, 7, "")
    assert text_segment.text == "text"

    text_segment.replace_substring(0, 4, "new'")
    assert text_segment.text == "new'"

    text_segment.replace_substring(3, 4, "\u2019")
    assert text_segment.text == "new\u2019"

    text_segment.replace_substring(0, 0, "prefix ")
    assert text_segment.text == "prefix new\u2019"

    text_segment.replace_substring(0, 0, "")
    assert text_segment.text == "prefix new\u2019"

    text_segment.replace_substring(11, 11, " suffix")
    assert text_segment.text == "prefix new\u2019 suffix"

    text_segment.replace_substring(6, 6, "-")
    assert text_segment.text == "prefix- new\u2019 suffix"
