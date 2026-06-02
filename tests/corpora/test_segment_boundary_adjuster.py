from machine.corpora.segment_boundary_adjuster import SegmentBoundaryAdjuster, TokenRejoiner


def test_prohibited_sentence_starting_characters() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # Second segment starts with space
    segment, next_segment = adjuster.adjust_segment_pair_boundary("In the beginning God created", " the heavens")
    assert segment == "In the beginning God created "
    assert next_segment == "the heavens"

    # Second segment starts with comma
    segment, next_segment = adjuster.adjust_segment_pair_boundary("He took the bread", ", blessed it")
    assert segment == "He took the bread, "
    assert next_segment == "blessed it"

    # Second segment starts with semicolon
    segment, next_segment = adjuster.adjust_segment_pair_boundary("first, apostles", "; second prophets")
    assert segment == "first, apostles; "
    assert next_segment == "second prophets"

    # Second segment starts with colon
    segment, next_segment = adjuster.adjust_segment_pair_boundary("He taught them, saying", ": Blessed are the")
    assert segment == "He taught them, saying: "
    assert next_segment == "Blessed are the"

    # Second segment starts with period
    segment, next_segment = adjuster.adjust_segment_pair_boundary("what belongs to God", ". They will only")
    assert segment == "what belongs to God. "
    assert next_segment == "They will only"

    # Second segment starts with exclamation mark
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "You hypocrite", "! First take the beam out of your own eye"
    )
    assert segment == "You hypocrite! "
    assert next_segment == "First take the beam out of your own eye"

    # Second segment starts with question mark
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Why do you worry about clothes", "? See how the flowers"
    )
    assert segment == "Why do you worry about clothes? "
    assert next_segment == "See how the flowers"

    # Second segment starts with closing parenthesis
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Simon (who is called Peter", ") and his brother Andrew"
    )
    assert segment == "Simon (who is called Peter) "
    assert next_segment == "and his brother Andrew"

    # Second segment starts with closing square bracket
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "manuscripts do not include John 7:53-8:11.", "] Then they all went"
    )
    assert segment == "manuscripts do not include John 7:53-8:11.] "
    assert next_segment == "Then they all went"

    # Second segment starts with closing double quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary("\u201cLord,", "\u201d he said")
    assert segment == "\u201cLord,\u201d "
    assert next_segment == "he said"

    # Second segment starts with closing single quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary("Your sins are forgiven,", "\u2019 or to say")
    assert segment == "Your sins are forgiven,\u2019 "
    assert next_segment == "or to say"

    # Second segment starts with multiple prohibited characters in a row
    segment, next_segment = adjuster.adjust_segment_pair_boundary("or to say", ", \u2018Get up and walk")
    assert segment == "or to say, "
    assert next_segment == "\u2018Get up and walk"


def test_prohibited_sentence_ending_characters() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # First segment ends with opening parenthesis
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "They said, \u201cRabbi\u2019 (", "which means \u201cTeacher\u201d)"
    )
    assert segment == "They said, \u201cRabbi\u2019 "
    assert next_segment == "(which means \u201cTeacher\u201d)"

    # First segment ends with opening square bracket
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "from Galilee!\u201d [", "The most ancient Greek manuscripts"
    )
    assert segment == "from Galilee!\u201d "
    assert next_segment == "[The most ancient Greek manuscripts"

    # First segment ends with double guillemet
    segment, next_segment = adjuster.adjust_segment_pair_boundary("Il dit \u00ab", "Venez à moi")
    assert segment == "Il dit "
    assert next_segment == "\u00abVenez à moi"

    # First segment ends with single guillemet
    segment, next_segment = adjuster.adjust_segment_pair_boundary("Jésus s'écria \u2039", "Père pourquoi")
    assert segment == "Jésus s'écria "
    assert next_segment == "\u2039Père pourquoi"

    # First segment ends with opening double quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Knowing their thoughts, Jesus said, \u201c", "Why do you"
    )
    assert segment == "Knowing their thoughts, Jesus said, "
    assert next_segment == "\u201cWhy do you"

    # First segment ends with opening single quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary("or to say, \u2018", "Get up and walk\u2019?")
    assert segment == "or to say, "
    assert next_segment == "\u2018Get up and walk\u2019?"

    # First segment ends with multiple prohibited characters in a row
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "This is what the Lord says: \u201c\u2018", "I remember the devotion of your youth"
    )
    assert segment == "This is what the Lord says: "
    assert next_segment == "\u201c\u2018I remember the devotion of your youth"


def test_late_sentence_starts() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # Sentence starts one word late
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "In the beginning God created the heavens and the earth. And ", "the earth was without form"
    )
    assert segment == "In the beginning God created the heavens and the earth. "
    assert next_segment == "And the earth was without form"

    # Sentence starts two words late
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "In the beginning God created the heavens and the earth. And the ", "earth was without form"
    )
    assert segment == "In the beginning God created the heavens and the earth. "
    assert next_segment == "And the earth was without form"

    # Sentence starts three words late
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "In the beginning God created the heavens and the earth. And the earth", "was without form"
    )
    assert segment == "In the beginning God created the heavens and the earth. "
    assert next_segment == "And the earth was without form"

    # Two-word capitalized phrase
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "are possible with God.\u201d Then Peter answered ", "and said to him,"
    )
    assert segment == "are possible with God.\u201d "
    assert next_segment == "Then Peter answered and said to him,"

    # Doesn't apply to uncapitalized word
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "In the beginning God created the heavens and the earth. and the earth", "was without form"
    )
    assert segment == "In the beginning God created the heavens and the earth. and the earth"
    assert next_segment == "was without form"


def test_early_sentence_starts() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # Sentence starts one word early
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "And the earth was without form and ", "void. And darkness"
    )
    assert segment == "And the earth was without form and void. "
    assert next_segment == "And darkness"

    # Sentence starts two words early
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "And the earth was without form ", "and void. And darkness"
    )
    assert segment == "And the earth was without form and void. "
    assert next_segment == "And darkness"

    # Sentence starts three words early
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "And the earth was without ", "form and void. And darkness"
    )
    assert segment == "And the earth was without form and void. "
    assert next_segment == "And darkness"

    # Early sentence end with comma
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "They have ", "forsaken me, the source of living water."
    )
    assert segment == "They have forsaken me, "
    assert next_segment == "the source of living water."

    # Early sentence end with semicolon
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Your wickedness will ", "punish you; your backsliding will rebuke you"
    )
    assert segment == "Your wickedness will punish you; "
    assert next_segment == "your backsliding will rebuke you"

    # Early sentence end with exclamation
    segment, next_segment = adjuster.adjust_segment_pair_boundary("look at ", "the fields! They are ripe for harvest.")
    assert segment == "look at the fields! "
    assert next_segment == "They are ripe for harvest."

    # Early sentence end with question mark
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Where can you get this ", "living water? Are you greater than our father Jacob?"
    )
    assert segment == "Where can you get this living water? "
    assert next_segment == "Are you greater than our father Jacob?"

    # Early sentence end with closing parenthesis
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "This was before John was put ", "in prison) An argument developed"
    )
    assert segment == "This was before John was put in prison) "
    assert next_segment == "An argument developed"

    # Early sentence end with closing square bracket
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "manuscripts do not include ", "this passage] Then they all went"
    )
    assert segment == "manuscripts do not include this passage] "
    assert next_segment == "Then they all went"

    # Early sentence end with closing double quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "your testimony is ", "not valid\u201d Jesus answered,"
    )
    assert segment == "your testimony is not valid\u201d "
    assert next_segment == "Jesus answered,"

    # Early sentence end with closing single quote
    segment, next_segment = adjuster.adjust_segment_pair_boundary("bread from heaven ", "to eat.’ Jesus said to them,")
    assert segment == "bread from heaven to eat.’ "
    assert next_segment == "Jesus said to them,"

    # Early sentence end with multiple closing quotes
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "\u2018Make straight the way for ", "the Lord.\u2019\u201d Now the Pharisees"
    )
    assert segment == "\u2018Make straight the way for the Lord.\u2019\u201d "
    assert next_segment == "Now the Pharisees"


def test_multiple_adjustments() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # Second segment starts with a comma and the sentence ends late
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "He took the bread", ", and blessed it. Then he gave it to them"
    )
    assert segment == "He took the bread, and blessed it. "
    assert next_segment == "Then he gave it to them"

    # Second segment starts with a comma and the sentence ends early
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "When you are persecuted in one place", ", flee to another. Truly I tell you"
    )
    assert segment == "When you are persecuted in one place, flee to another. "
    assert next_segment == "Truly I tell you"

    # First segment ends with an opening double quote and the sentence starts late
    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "Jesus knew their thoughts. He said \u201c", "Why do you entertain evil thoughts in your hearts?"
    )
    assert segment == "Jesus knew their thoughts. "
    assert next_segment == "He said \u201cWhy do you entertain evil thoughts in your hearts?"


def test_multiple_segments() -> None:
    adjuster = SegmentBoundaryAdjuster()

    verses = ["Jesus said, \u201c", "Come unto me all who are weary", "; I will give you rest"]
    adjusted_verses = adjuster.adjust_segment_boundaries(verses)

    assert adjusted_verses == ["Jesus said, ", "\u201cCome unto me all who are weary; ", "I will give you rest"]


def test_tokenized_segments() -> None:
    adjuster = SegmentBoundaryAdjuster()

    # Second segment starts with comma
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        4, ["He", "took", "the", "bread", ",", "blessed", "it"]
    )
    assert adjusted_boundary == 5

    # Second segment starts with multiple prohibited characters in a row
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        4, ["I", "have", "no", "husband", ",", "\u201d", "she", "replied"]
    )
    assert adjusted_boundary == 6

    # First segment ends with opening double quote
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        8, ["Knowing", "their", "thoughts", ",", "Jesus", "said", ",", "\u201c", "Why", "do", "you"]
    )
    assert adjusted_boundary == 7

    # First segment ends with multiple prohibited characters in a row
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        9,
        [
            "This",
            "is",
            "what",
            "the",
            "Lord",
            "says",
            ":",
            "\u201c",
            "\u2018",
            "I",
            "remember",
            "the",
            "devotion",
            "of",
            "your",
            "youth",
        ],
    )
    assert adjusted_boundary == 7

    # Sentence starts three words late
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        14,
        [
            "In",
            "the",
            "beginning",
            "God",
            "created",
            "the",
            "heavens",
            "and",
            "the",
            "earth",
            ".",
            "And",
            "the",
            "earth",
            "was",
            "without",
            "form",
        ],
    )
    assert adjusted_boundary == 11

    # Early sentence end with question mark
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        6,
        [
            "Where",
            "can",
            "you",
            "get",
            "this",
            ",",
            "living",
            "water",
            "?",
            "Are",
            "you",
            "greater",
            "than",
            "our",
            "father",
            "Jacob",
            "?",
        ],
    )
    assert adjusted_boundary == 9

    # Early sentence end with multiple closing quotes
    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        6,
        [
            "\u2018",
            "Make",
            "straight",
            "the",
            "way",
            "for",
            "the",
            "Lord",
            ".",
            "\u2019",
            "\u201d",
            "Now",
            "the",
            "Pharisees",
        ],
    )
    assert adjusted_boundary == 11


def test_no_adjustment_for_correct_boundaries() -> None:
    adjuster = SegmentBoundaryAdjuster()

    segment, next_segment = adjuster.adjust_segment_pair_boundary(
        "In the beginning God created the heavens and the earth.", "And the earth was without form"
    )
    assert segment == "In the beginning God created the heavens and the earth."
    assert next_segment == "And the earth was without form"

    adjusted_segments = adjuster.adjust_segment_boundaries(
        [
            "\u201cWhy do you entertain evil thoughts in your hearts?",
            "Which is easier: to say, ",
            "\u2018Your sins are forgiven,\u2019",
            "or to say",
            "\u2018Get up and walk\u2019?",
        ]
    )
    assert adjusted_segments == [
        "\u201cWhy do you entertain evil thoughts in your hearts?",
        "Which is easier: to say, ",
        "\u2018Your sins are forgiven,\u2019",
        "or to say",
        "\u2018Get up and walk\u2019?",
    ]

    adjusted_boundary = adjuster.adjust_tokenized_segment_pair_boundaries(
        8,
        [
            "But",
            "go",
            "and",
            "learn",
            "what",
            "this",
            "means",
            ":",
            "\u2018",
            "I",
            "desire",
            "mercy",
            "not",
            "sacrifice",
            "\u2019",
        ],
    )
    assert adjusted_boundary == 8


def test_join_tokens() -> None:
    assert (
        TokenRejoiner.join_tokens(["He", "took", "the", "bread", ",", "blessed", "it"])
        == "He took the bread, blessed it "
    )

    assert TokenRejoiner.join_tokens(
        ["Knowing", "their", "thoughts", ",", "Jesus", "said", ",", "\u201c", "Why", "do", "you"]
    ) == ("Knowing their thoughts, Jesus said, \u201cWhy do you ")

    assert (
        TokenRejoiner.join_tokens(["Freely", "you", "have", "received", ";", "freely", "give", "."])
        == "Freely you have received; freely give. "
    )

    assert (
        TokenRejoiner.join_tokens(["Il", "dit", "\u00ab", "Venez", "\u00bb", ".", "Maintenant"])
        == "Il dit \u00abVenez\u00bb. Maintenant "
    )

    assert (
        TokenRejoiner.join_tokens(["Il", "dit", "<<", "Venez", ">>", ".", "Maintenant"])
        == "Il dit <<Venez>>. Maintenant "
    )

    assert (
        TokenRejoiner.join_tokens(
            ["J\u00e9sus", "s'\u00e9cria", "\u2039", "P\u00e8re", "!", "\u203a", ",", "puis", "partit"]
        )
        == "J\u00e9sus s'\u00e9cria \u2039P\u00e8re!\u203a, puis partit "
    )


def test_add_token_to_joined_text() -> None:
    rejoiner = TokenRejoiner()

    assert rejoiner.add_token_to_joined_text("Knowing") == "Knowing"
    assert rejoiner.add_token_to_joined_text("their") == "Knowing their"
    assert rejoiner.add_token_to_joined_text("thoughts") == "Knowing their thoughts"
    assert rejoiner.add_token_to_joined_text(",") == "Knowing their thoughts,"
    assert rejoiner.add_token_to_joined_text("Jesus") == "Knowing their thoughts, Jesus"
    assert rejoiner.add_token_to_joined_text("said") == "Knowing their thoughts, Jesus said"
    assert rejoiner.add_token_to_joined_text(",") == "Knowing their thoughts, Jesus said,"
    assert rejoiner.add_token_to_joined_text("\u201c") == "Knowing their thoughts, Jesus said, \u201c"
    assert rejoiner.add_token_to_joined_text("Why") == "Knowing their thoughts, Jesus said, \u201cWhy"
    assert rejoiner.add_token_to_joined_text("do") == "Knowing their thoughts, Jesus said, \u201cWhy do"
    assert rejoiner.add_token_to_joined_text("you") == "Knowing their thoughts, Jesus said, \u201cWhy do you"
