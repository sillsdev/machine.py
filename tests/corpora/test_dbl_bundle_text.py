from machine.scripture import VerseRef

from .dbl_bundle_test_environment import DblBundleTestEnvironment


def test_get_segments_nonempty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 48

        assert segments[0].segment_ref == VerseRef.from_string("MAT 1:1", env.corpus.versification)
        assert (
            segments[0].segment[0]
            == "This is the record of the ancestors of Jesus the Messiah, the descendant of King David and of Abraham, "
            "from whom all we Jews have descended."
        )

        assert segments[1].segment_ref == VerseRef.from_string("MAT 1:2", env.corpus.versification)
        assert (
            segments[1].segment[0]
            == "Abraham was the father of Isaac. Isaac was the father of Jacob. Jacob was the father of Judah and "
            "Judah's older and younger brothers."
        )

        assert segments[25].segment_ref == VerseRef.from_string("MAT 2:1", env.corpus.versification)
        assert (
            segments[25].segment[0]
            == "Jesus was born in Bethlehem town in Judea province during the time [MTY] that King Herod the Great "
            "ruled there. Some time after Jesus was born, some men who studied the stars and who lived in a country "
            "east of Judea came to Jerusalem city."
        )

        assert segments[36].segment_ref == VerseRef.from_string("MAT 2:12", env.corpus.versification)
        assert (
            segments[36].segment[0]
            == "Because God knew that King Herod planned to kill Jesus, in a dream the men who studied the stars were "
            "warned {he warned the men who studied the stars} that they should not return to King Herod. So they "
            "returned to their country, but instead of traveling back on the same road, they went on a different road."
        )

        assert segments[39].segment_ref == VerseRef.from_string("MAT 2:15", env.corpus.versification)
        assert (
            segments[39].segment[0]
            == "They stayed there until King Herod died, and then they left Egypt. By doing that, it was {they} "
            "fulfilled what the prophet Hosea wrote, which had been said by the Lord {which the Lord had said}, I have "
            "told my son to come out of Egypt."
        )

        assert segments[45].segment_ref == VerseRef.from_string("MAT 2:21", env.corpus.versification)
        assert segments[45].segment[0] == "So Joseph took the child and his mother, and they went back to Israel."

        assert segments[46].segment_ref == VerseRef.from_string("MAT 2:22", env.corpus.versification)
        assert (
            segments[46].segment[0]
            == "When Joseph heard that Archaelaus now ruled in Judea district instead of his father, King Herod the "
            "Great, he was afraid to go there. Because he was warned {God warned Joseph} in a dream that it was still "
            "dangerous for them to live in Judea, he and Mary and Jesus went to Galilee District"
        )


def test_get_segments_sentence_start() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MAT")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 48

        assert segments[38].segment_ref == VerseRef.from_string("MAT 2:14", env.corpus.versification)
        assert (
            segments[38].segment[0]
            == "So Joseph got up, he took the child and his mother that night, and they fled to Egypt."
        )
        assert segments[38].is_sentence_start

        assert segments[46].segment_ref == VerseRef.from_string("MAT 2:22", env.corpus.versification)
        assert (
            segments[46].segment[0]
            == "When Joseph heard that Archaelaus now ruled in Judea district instead of his father, King Herod the "
            "Great, he was afraid to go there. Because he was warned {God warned Joseph} in a dream that it was still "
            "dangerous for them to live in Judea, he and Mary and Jesus went to Galilee District"
        )
        assert segments[46].is_sentence_start

        assert segments[47].segment_ref == VerseRef.from_string("MAT 2:23", env.corpus.versification)
        assert (
            segments[47].segment[0]
            == "to the town called Nazareth to live there. The result was that what had been said by the ancient "
            "prophets {what the ancient prophets had said} about the Messiah, that he would be called {people would "
            "call him} a Nazareth-man, was fulfilled {came true}."
        )
        assert not segments[47].is_sentence_start


def test_get_segments_empty_text() -> None:
    with DblBundleTestEnvironment() as env:
        text = env.corpus.get_text("MRK")
        assert text is not None
        segments = list(text.get_segments())

        assert len(segments) == 0
