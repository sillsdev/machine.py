from pytest import approx

from machine.corpora.analysis import (
    QuotationMarkDirection,
    QuotationMarkMetadata,
    QuotationMarkTabulator,
    QuoteConvention,
    QuoteConventionSet,
    SingleLevelQuoteConvention,
    TextSegment,
)


def test_quote_regexes() -> None:
    empty_quote_convention_set = QuoteConventionSet([])
    assert empty_quote_convention_set._opening_quotation_mark_regex.pattern == r""
    assert empty_quote_convention_set._closing_quotation_mark_regex.pattern == r""
    assert empty_quote_convention_set._all_quotation_mark_regex.pattern == r""

    quote_convention_set_with_empty_conventions = QuoteConventionSet(
        [QuoteConvention("empty convention 1", []), QuoteConvention("empty convention 2", [])]
    )
    assert quote_convention_set_with_empty_conventions._opening_quotation_mark_regex.pattern == r""
    assert quote_convention_set_with_empty_conventions._closing_quotation_mark_regex.pattern == r""
    assert quote_convention_set_with_empty_conventions._all_quotation_mark_regex.pattern == r""

    standard_english_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "standard_english",
                [
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            )
        ]
    )
    assert standard_english_quote_convention_set._opening_quotation_mark_regex.pattern == r"[‘“]"
    assert standard_english_quote_convention_set._closing_quotation_mark_regex.pattern == r"[’”]"
    assert standard_english_quote_convention_set._all_quotation_mark_regex.pattern == r"[‘’“”]"

    western_european_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "western_european",
                [
                    SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            ),
        ]
    )
    assert western_european_quote_convention_set._opening_quotation_mark_regex.pattern == r"[«‘“]"
    assert western_european_quote_convention_set._closing_quotation_mark_regex.pattern == r"[»’”]"
    assert western_european_quote_convention_set._all_quotation_mark_regex.pattern == r"[«»‘’“”]"

    multiple_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "standard_english",
                [
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            ),
            QuoteConvention(
                "typewriter_french",
                [
                    SingleLevelQuoteConvention("<<", ">>"),
                    SingleLevelQuoteConvention("<", ">"),
                    SingleLevelQuoteConvention("<<", ">>"),
                    SingleLevelQuoteConvention("<", ">"),
                ],
            ),
            QuoteConvention(
                "standard_french",
                [
                    SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                    SingleLevelQuoteConvention("\u2039", "\u203a"),
                    SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                    SingleLevelQuoteConvention("\u2039", "\u203a"),
                ],
            ),
        ]
    )
    assert multiple_quote_convention_set._opening_quotation_mark_regex.pattern == r"[<<<«‘“‹]"
    assert multiple_quote_convention_set._closing_quotation_mark_regex.pattern == r"[>>>»’”›]"
    assert multiple_quote_convention_set._all_quotation_mark_regex.pattern == r"[<<<>>>«»‘’“”‹›]"


def test_quotation_mark_pair_map() -> None:
    empty_quote_convention_set = QuoteConventionSet([])
    assert empty_quote_convention_set.closing_marks_by_opening_mark == {}
    assert empty_quote_convention_set.opening_marks_by_closing_mark == {}

    quote_convention_set_with_empty_conventions = QuoteConventionSet(
        [QuoteConvention("empty convention 1", []), QuoteConvention("empty convention 2", [])]
    )
    assert quote_convention_set_with_empty_conventions.closing_marks_by_opening_mark == {}
    assert quote_convention_set_with_empty_conventions.opening_marks_by_closing_mark == {}

    standard_english_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "standard_english",
                [
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            )
        ]
    )
    assert standard_english_quote_convention_set.closing_marks_by_opening_mark == {"‘": {"’"}, "“": {"”"}}
    assert standard_english_quote_convention_set.opening_marks_by_closing_mark == {"’": {"‘"}, "”": {"“"}}

    western_european_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "western_european",
                [
                    SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            ),
        ]
    )
    assert western_european_quote_convention_set.closing_marks_by_opening_mark == {"‘": {"’"}, "“": {"”"}, "«": {"»"}}
    assert western_european_quote_convention_set.opening_marks_by_closing_mark == {"’": {"‘"}, "”": {"“"}, "»": {"«"}}

    multiple_quote_convention_set = QuoteConventionSet(
        [
            QuoteConvention(
                "standard_english",
                [
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                    SingleLevelQuoteConvention("\u201c", "\u201d"),
                    SingleLevelQuoteConvention("\u2018", "\u2019"),
                ],
            ),
            QuoteConvention(
                "central_european",
                [
                    SingleLevelQuoteConvention("\u201e", "\u201c"),
                    SingleLevelQuoteConvention("\u201a", "\u2018"),
                    SingleLevelQuoteConvention("\u201e", "\u201c"),
                    SingleLevelQuoteConvention("\u201a", "\u2018"),
                ],
            ),
            QuoteConvention(
                "standard_swedish",
                [
                    SingleLevelQuoteConvention("\u201d", "\u201d"),
                    SingleLevelQuoteConvention("\u2019", "\u2019"),
                    SingleLevelQuoteConvention("\u201d", "\u201d"),
                    SingleLevelQuoteConvention("\u2019", "\u2019"),
                ],
            ),
        ]
    )
    assert multiple_quote_convention_set.closing_marks_by_opening_mark == {
        "‘": {"’"},
        "“": {"”"},
        "„": {"“"},
        "‚": {"‘"},
        "”": {"”"},
        "’": {"’"},
    }
    assert multiple_quote_convention_set.opening_marks_by_closing_mark == {
        "’": {"‘", "’"},
        "”": {"“", "”"},
        "“": {"„"},
        "‘": {"‚"},
    }


def test_get_quote_convention_by_name() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )
    multiple_quote_convention_set = QuoteConventionSet(
        [standard_english_quote_convention, central_european_quote_convention, standard_swedish_quote_convention]
    )

    assert (
        multiple_quote_convention_set.get_quote_convention_by_name("standard_english")
        == standard_english_quote_convention
    )
    assert (
        multiple_quote_convention_set.get_quote_convention_by_name("central_european")
        == central_european_quote_convention
    )
    assert (
        multiple_quote_convention_set.get_quote_convention_by_name("standard_swedish")
        == standard_swedish_quote_convention
    )
    assert multiple_quote_convention_set.get_quote_convention_by_name("undefined convention") is None


def test_get_all_quote_convention_names() -> None:
    assert QuoteConventionSet([]).get_all_quote_convention_names() == []
    assert QuoteConventionSet([QuoteConvention("conv", [])]).get_all_quote_convention_names() == ["conv"]
    assert QuoteConventionSet(
        [QuoteConvention("conv1", []), QuoteConvention("conv2", [])]
    ).get_all_quote_convention_names() == ["conv1", "conv2"]
    assert QuoteConventionSet(
        [QuoteConvention("conv2", []), QuoteConvention("conv1", [])]
    ).get_all_quote_convention_names() == ["conv1", "conv2"]


def test_get_possible_opening_marks() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.get_possible_opening_marks() == ["‘", "“"]

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.get_possible_opening_marks() == ["‚", "„"]

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.get_possible_opening_marks() == ["’", "”"]

    multiple_quote_convention_set = QuoteConventionSet(
        [standard_english_quote_convention, central_european_quote_convention, standard_swedish_quote_convention]
    )
    assert multiple_quote_convention_set.get_possible_opening_marks() == ["‘", "’", "‚", "“", "”", "„"]


def test_get_possible_closing_marks() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.get_possible_closing_marks() == ["’", "”"]

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.get_possible_closing_marks() == ["‘", "“"]

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.get_possible_closing_marks() == ["’", "”"]

    multiple_quote_convention_set = QuoteConventionSet(
        [standard_english_quote_convention, central_european_quote_convention, standard_swedish_quote_convention]
    )
    assert multiple_quote_convention_set.get_possible_closing_marks() == ["‘", "’", "“", "”"]


def test_is_opening_quotation_mark() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_french_quote_convention: QuoteConvention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.is_valid_opening_quotation_mark("‘")
    assert standard_english_quote_convention_set.is_valid_opening_quotation_mark("“")
    assert not standard_english_quote_convention_set.is_valid_opening_quotation_mark("”")
    assert not standard_english_quote_convention_set.is_valid_opening_quotation_mark("’")
    assert not standard_english_quote_convention_set.is_valid_opening_quotation_mark("")
    assert not standard_english_quote_convention_set.is_valid_opening_quotation_mark("‘“")

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.is_valid_opening_quotation_mark("‚")
    assert central_european_quote_convention_set.is_valid_opening_quotation_mark("„")
    assert not central_european_quote_convention_set.is_valid_opening_quotation_mark("‘")
    assert not central_european_quote_convention_set.is_valid_opening_quotation_mark("“")

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.is_valid_opening_quotation_mark("’")
    assert standard_swedish_quote_convention_set.is_valid_opening_quotation_mark("”")

    standard_french_quote_convention_set = QuoteConventionSet([standard_french_quote_convention])
    assert standard_french_quote_convention_set.is_valid_opening_quotation_mark("«")
    assert standard_french_quote_convention_set.is_valid_opening_quotation_mark("‹")
    assert not standard_french_quote_convention_set.is_valid_opening_quotation_mark("»")
    assert not standard_french_quote_convention_set.is_valid_opening_quotation_mark("›")

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            central_european_quote_convention,
            standard_swedish_quote_convention,
            standard_french_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.get_possible_opening_marks() == ["«", "‘", "’", "‚", "“", "”", "„", "‹"]
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("‘")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("’")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("‚")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("“")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("”")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("„")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("«")
    assert multiple_quote_convention_set.is_valid_opening_quotation_mark("‹")
    assert not multiple_quote_convention_set.is_valid_opening_quotation_mark("»")
    assert not multiple_quote_convention_set.is_valid_opening_quotation_mark("›")


def test_is_closing_quotation_mark() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_french_quote_convention: QuoteConvention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.is_valid_closing_quotation_mark("”")
    assert standard_english_quote_convention_set.is_valid_closing_quotation_mark("’")
    assert not standard_english_quote_convention_set.is_valid_closing_quotation_mark("‘")
    assert not standard_english_quote_convention_set.is_valid_closing_quotation_mark("“")
    assert not standard_english_quote_convention_set.is_valid_closing_quotation_mark("")
    assert not standard_english_quote_convention_set.is_valid_closing_quotation_mark("”’")

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.is_valid_closing_quotation_mark("‘")
    assert central_european_quote_convention_set.is_valid_closing_quotation_mark("“")
    assert not central_european_quote_convention_set.is_valid_closing_quotation_mark("„")
    assert not central_european_quote_convention_set.is_valid_closing_quotation_mark("‚")

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.is_valid_closing_quotation_mark("’")
    assert standard_swedish_quote_convention_set.is_valid_closing_quotation_mark("”")

    standard_french_quote_convention_set = QuoteConventionSet([standard_french_quote_convention])
    assert standard_french_quote_convention_set.is_valid_closing_quotation_mark("»")
    assert standard_french_quote_convention_set.is_valid_closing_quotation_mark("›")
    assert not standard_french_quote_convention_set.is_valid_closing_quotation_mark("«")
    assert not standard_french_quote_convention_set.is_valid_closing_quotation_mark("‹")

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            central_european_quote_convention,
            standard_swedish_quote_convention,
            standard_french_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.get_possible_closing_marks() == ["»", "‘", "’", "“", "”", "›"]
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("‘")
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("’")
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("“")
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("”")
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("»")
    assert multiple_quote_convention_set.is_valid_closing_quotation_mark("›")
    assert not multiple_quote_convention_set.is_valid_closing_quotation_mark("«")
    assert not multiple_quote_convention_set.is_valid_closing_quotation_mark("‹")


def test_are_marks_a_valid_pair() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_french_quote_convention: QuoteConvention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.marks_are_a_valid_pair("“", "”")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("”", "“")
    assert standard_english_quote_convention_set.marks_are_a_valid_pair("‘", "’")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("’", "‘")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("‘", "”")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("‘", "”")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("‘", "")
    assert not standard_english_quote_convention_set.marks_are_a_valid_pair("", "")

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.marks_are_a_valid_pair("„", "“")
    assert central_european_quote_convention_set.marks_are_a_valid_pair("‚", "‘")
    assert not central_european_quote_convention_set.marks_are_a_valid_pair("“", "„")
    assert not central_european_quote_convention_set.marks_are_a_valid_pair("’", "‚")
    assert not central_european_quote_convention_set.marks_are_a_valid_pair("‚", "“")
    assert not central_european_quote_convention_set.marks_are_a_valid_pair("‚", "’")

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.marks_are_a_valid_pair("”", "”")
    assert standard_swedish_quote_convention_set.marks_are_a_valid_pair("’", "’")
    assert not standard_swedish_quote_convention_set.marks_are_a_valid_pair("”", "’")
    assert not standard_swedish_quote_convention_set.marks_are_a_valid_pair("’", "”")

    standard_french_quote_convention_set = QuoteConventionSet([standard_french_quote_convention])
    assert standard_french_quote_convention_set.marks_are_a_valid_pair("«", "»")
    assert standard_french_quote_convention_set.marks_are_a_valid_pair("‹", "›")
    assert not standard_french_quote_convention_set.marks_are_a_valid_pair("«", "›")
    assert not standard_french_quote_convention_set.marks_are_a_valid_pair("‹", "»")

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            central_european_quote_convention,
            standard_swedish_quote_convention,
            standard_french_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.marks_are_a_valid_pair("“", "”")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("‘", "’")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("„", "“")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("‚", "‘")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("”", "”")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("’", "’")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("«", "»")
    assert multiple_quote_convention_set.marks_are_a_valid_pair("‹", "›")
    assert not multiple_quote_convention_set.marks_are_a_valid_pair("‹", "»")
    assert not multiple_quote_convention_set.marks_are_a_valid_pair("‹", "”")
    assert not multiple_quote_convention_set.marks_are_a_valid_pair("„", "”")
    assert not multiple_quote_convention_set.marks_are_a_valid_pair("’", "‘")


def test_is_quotation_mark_direction_ambiguous() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    typewriter_english_quote_convention: QuoteConvention = QuoteConvention(
        "typewriter_english",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    eastern_european_quote_convention = QuoteConvention(
        "eastern_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201d"),
            SingleLevelQuoteConvention("\u201a", "\u2019"),
            SingleLevelQuoteConvention("\u201e", "\u201d"),
            SingleLevelQuoteConvention("\u201a", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert not standard_english_quote_convention_set.is_quotation_mark_direction_ambiguous("“")
    assert not standard_english_quote_convention_set.is_quotation_mark_direction_ambiguous("”")
    assert not standard_english_quote_convention_set.is_quotation_mark_direction_ambiguous("‘")
    assert not standard_english_quote_convention_set.is_quotation_mark_direction_ambiguous("’")
    assert not standard_english_quote_convention_set.is_quotation_mark_direction_ambiguous('"')

    typewriter_english_quote_convention_set = QuoteConventionSet([typewriter_english_quote_convention])
    assert typewriter_english_quote_convention_set.is_quotation_mark_direction_ambiguous('"')
    assert typewriter_english_quote_convention_set.is_quotation_mark_direction_ambiguous("'")
    assert not typewriter_english_quote_convention_set.is_quotation_mark_direction_ambiguous("‘")
    assert not typewriter_english_quote_convention_set.is_quotation_mark_direction_ambiguous("’")
    assert not typewriter_english_quote_convention_set.is_quotation_mark_direction_ambiguous("«")

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert not central_european_quote_convention_set.is_quotation_mark_direction_ambiguous("“")
    assert not central_european_quote_convention_set.is_quotation_mark_direction_ambiguous("„")
    assert not central_european_quote_convention_set.is_quotation_mark_direction_ambiguous("‘")
    assert not central_european_quote_convention_set.is_quotation_mark_direction_ambiguous("‚")

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.is_quotation_mark_direction_ambiguous("”")
    assert standard_swedish_quote_convention_set.is_quotation_mark_direction_ambiguous("’")

    eastern_european_quote_convention_set = QuoteConventionSet([eastern_european_quote_convention])
    assert not eastern_european_quote_convention_set.is_quotation_mark_direction_ambiguous("”")
    assert not eastern_european_quote_convention_set.is_quotation_mark_direction_ambiguous("„")
    assert not eastern_european_quote_convention_set.is_quotation_mark_direction_ambiguous("’")
    assert not eastern_european_quote_convention_set.is_quotation_mark_direction_ambiguous("‚")

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            typewriter_english_quote_convention,
            central_european_quote_convention,
            standard_swedish_quote_convention,
            eastern_european_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.is_quotation_mark_direction_ambiguous('"')
    assert multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("'")
    assert multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("”")
    assert multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("’")
    assert not multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("„")
    assert not multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("‚")

    # these are unambiguous because they are never the opening and closing in the same convention
    assert not multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("“")
    assert not multiple_quote_convention_set.is_quotation_mark_direction_ambiguous("‘")


def test_get_possible_paired_quotation_marks() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    central_european_quote_convention: QuoteConvention = QuoteConvention(
        "central_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
            SingleLevelQuoteConvention("\u201e", "\u201c"),
            SingleLevelQuoteConvention("\u201a", "\u2018"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    eastern_european_quote_convention = QuoteConvention(
        "eastern_european",
        [
            SingleLevelQuoteConvention("\u201e", "\u201d"),
            SingleLevelQuoteConvention("\u201a", "\u2019"),
            SingleLevelQuoteConvention("\u201e", "\u201d"),
            SingleLevelQuoteConvention("\u201a", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.get_possible_paired_quotation_marks("“") == {"”"}
    assert standard_english_quote_convention_set.get_possible_paired_quotation_marks("”") == {"“"}
    assert standard_english_quote_convention_set.get_possible_paired_quotation_marks("‘") == {"’"}
    assert standard_english_quote_convention_set.get_possible_paired_quotation_marks("’") == {"‘"}

    central_european_quote_convention_set = QuoteConventionSet([central_european_quote_convention])
    assert central_european_quote_convention_set.get_possible_paired_quotation_marks("„") == {"“"}
    assert central_european_quote_convention_set.get_possible_paired_quotation_marks("“") == {"„"}
    assert central_european_quote_convention_set.get_possible_paired_quotation_marks("‚") == {"‘"}
    assert central_european_quote_convention_set.get_possible_paired_quotation_marks("‘") == {"‚"}

    standard_swedish_quote_convention_set = QuoteConventionSet([standard_swedish_quote_convention])
    assert standard_swedish_quote_convention_set.get_possible_paired_quotation_marks("”") == {"”"}
    assert standard_swedish_quote_convention_set.get_possible_paired_quotation_marks("’") == {"’"}

    eastern_european_quote_convention_set = QuoteConventionSet([eastern_european_quote_convention])
    assert eastern_european_quote_convention_set.get_possible_paired_quotation_marks("„") == {"”"}
    assert eastern_european_quote_convention_set.get_possible_paired_quotation_marks("”") == {"„"}
    assert eastern_european_quote_convention_set.get_possible_paired_quotation_marks("‚") == {"’"}
    assert eastern_european_quote_convention_set.get_possible_paired_quotation_marks("’") == {"‚"}

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            central_european_quote_convention,
            standard_swedish_quote_convention,
            eastern_european_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("“") == {"”", "„"}
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("”") == {"“", "”", "„"}
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("‘") == {"’", "‚"}
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("’") == {"‘", "’", "‚"}
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("„") == {"“", "”"}
    assert multiple_quote_convention_set.get_possible_paired_quotation_marks("‚") == {"‘", "’"}


def test_get_possible_depths() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    british_english_quote_convention: QuoteConvention = QuoteConvention(
        "british_english",
        [
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
        ],
    )

    normalized_western_european_quote_convention = QuoteConvention(
        "western_european_normalized",
        [
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention('"', '"'),
            SingleLevelQuoteConvention("'", "'"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.OPENING) == {1, 3}
    assert standard_english_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.CLOSING) == set()
    assert standard_english_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.CLOSING) == {1, 3}
    assert standard_english_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.OPENING) == set()
    assert standard_english_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.OPENING) == {2, 4}
    assert standard_english_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.CLOSING) == set()
    assert standard_english_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.CLOSING) == {2, 4}
    assert standard_english_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.OPENING) == set()
    assert standard_english_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.OPENING) == set()
    assert standard_english_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.CLOSING) == set()
    assert standard_english_quote_convention_set.get_possible_depths('"', QuotationMarkDirection.OPENING) == set()
    assert standard_english_quote_convention_set.get_possible_depths('"', QuotationMarkDirection.CLOSING) == set()

    british_english_quote_convention_set = QuoteConventionSet([british_english_quote_convention])
    assert british_english_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.OPENING) == {1, 3}
    assert british_english_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.CLOSING) == set()
    assert british_english_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.CLOSING) == {1, 3}
    assert british_english_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.OPENING) == set()
    assert british_english_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.OPENING) == {2, 4}
    assert british_english_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.CLOSING) == set()
    assert british_english_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.CLOSING) == {2, 4}
    assert british_english_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.OPENING) == set()
    assert british_english_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.OPENING) == set()
    assert british_english_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.CLOSING) == set()
    assert british_english_quote_convention_set.get_possible_depths("'", QuotationMarkDirection.OPENING) == set()
    assert british_english_quote_convention_set.get_possible_depths("'", QuotationMarkDirection.CLOSING) == set()

    normalized_western_european_quote_convention_set = QuoteConventionSet(
        [normalized_western_european_quote_convention]
    )
    assert normalized_western_european_quote_convention_set.get_possible_depths(
        '"', QuotationMarkDirection.OPENING
    ) == {1, 2}
    assert normalized_western_european_quote_convention_set.get_possible_depths(
        '"', QuotationMarkDirection.CLOSING
    ) == {1, 2}
    assert normalized_western_european_quote_convention_set.get_possible_depths(
        "'", QuotationMarkDirection.OPENING
    ) == {3}
    assert normalized_western_european_quote_convention_set.get_possible_depths(
        "'", QuotationMarkDirection.CLOSING
    ) == {3}
    assert (
        normalized_western_european_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.OPENING)
        == set()
    )
    assert (
        normalized_western_european_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.CLOSING)
        == set()
    )

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            british_english_quote_convention,
            normalized_western_european_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.OPENING) == {1, 2, 3, 4}
    assert multiple_quote_convention_set.get_possible_depths("\u201c", QuotationMarkDirection.CLOSING) == set()
    assert multiple_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.CLOSING) == {1, 2, 3, 4}
    assert multiple_quote_convention_set.get_possible_depths("\u201d", QuotationMarkDirection.OPENING) == set()
    assert multiple_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.OPENING) == {1, 2, 3, 4}
    assert multiple_quote_convention_set.get_possible_depths("\u2018", QuotationMarkDirection.CLOSING) == set()
    assert multiple_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.CLOSING) == {1, 2, 3, 4}
    assert multiple_quote_convention_set.get_possible_depths("\u2019", QuotationMarkDirection.OPENING) == set()
    assert multiple_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.OPENING) == set()
    assert multiple_quote_convention_set.get_possible_depths("\u201e", QuotationMarkDirection.CLOSING) == set()
    assert multiple_quote_convention_set.get_possible_depths('"', QuotationMarkDirection.OPENING) == {1, 2}
    assert multiple_quote_convention_set.get_possible_depths('"', QuotationMarkDirection.CLOSING) == {1, 2}
    assert multiple_quote_convention_set.get_possible_depths("'", QuotationMarkDirection.OPENING) == {3}
    assert multiple_quote_convention_set.get_possible_depths("'", QuotationMarkDirection.CLOSING) == {3}


def test_does_metadata_match_quotation_mark() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 1, QuotationMarkDirection.OPENING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 3, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 2, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 4, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 1, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 2, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 3, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201c", 4, QuotationMarkDirection.CLOSING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 1, QuotationMarkDirection.CLOSING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 3, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 2, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 4, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 1, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 2, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 3, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201d", 4, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 1, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 3, QuotationMarkDirection.OPENING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 2, QuotationMarkDirection.OPENING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 4, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 1, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 2, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 3, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2018", 4, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 1, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 3, QuotationMarkDirection.CLOSING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 2, QuotationMarkDirection.CLOSING
    )
    assert standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 4, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 1, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 2, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 3, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u2019", 4, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 1, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 1, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 2, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 2, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 3, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 3, QuotationMarkDirection.CLOSING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 4, QuotationMarkDirection.OPENING
    )
    assert not standard_english_quote_convention_set.metadata_matches_quotation_mark(
        "\u201e", 4, QuotationMarkDirection.CLOSING
    )


def test_filter_to_compatible_quote_conventions() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    standard_french_quote_convention: QuoteConvention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    western_european_quote_convention: QuoteConvention = QuoteConvention(
        "western_european",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    standard_swedish_quote_convention: QuoteConvention = QuoteConvention(
        "standard_swedish",
        [
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
            SingleLevelQuoteConvention("\u201d", "\u201d"),
            SingleLevelQuoteConvention("\u2019", "\u2019"),
        ],
    )

    standard_english_quote_convention_set = QuoteConventionSet([standard_english_quote_convention])
    assert standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c"], ["\u201d"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c", "\u2018"], ["\u201d", "\u2019"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c", "\u2018"], ["\u201d"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c"], ["\u201d", "\u2019"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u2018"], ["\u201d"]
        ).get_all_quote_convention_names()
        == []
    )
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u201c"], ["\u2019"]
        ).get_all_quote_convention_names()
        == []
    )
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u201d"], ["\u201c"]
        ).get_all_quote_convention_names()
        == []
    )
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u201c", "\u201d"], ["\u201d"]
        ).get_all_quote_convention_names()
        == []
    )
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u201c", "\u201e"], ["\u201d"]
        ).get_all_quote_convention_names()
        == []
    )
    assert (
        standard_english_quote_convention_set.filter_to_compatible_quote_conventions(
            [], []
        ).get_all_quote_convention_names()
        == []
    )

    multiple_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            standard_french_quote_convention,
            western_european_quote_convention,
            standard_swedish_quote_convention,
        ]
    )
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c"], ["\u201d"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201c", "\u2018"], ["\u201d", "\u2019"]
    ).get_all_quote_convention_names() == ["standard_english"]
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u201d"], ["\u201d"]
    ).get_all_quote_convention_names() == ["standard_swedish"]
    assert (
        multiple_quote_convention_set.filter_to_compatible_quote_conventions(
            ["\u201c"], ["\u201c"]
        ).get_all_quote_convention_names()
        == []
    )
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u00ab"], ["\u00bb"]
    ).get_all_quote_convention_names() == ["standard_french", "western_european"]
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u00ab", "\u2039"], ["\u00bb"]
    ).get_all_quote_convention_names() == ["standard_french"]
    assert multiple_quote_convention_set.filter_to_compatible_quote_conventions(
        ["\u00ab"], ["\u00bb", "\u201d"]
    ).get_all_quote_convention_names() == ["western_european"]
    assert (
        multiple_quote_convention_set.filter_to_compatible_quote_conventions([], []).get_all_quote_convention_names()
        == []
    )


def test_find_most_similar_convention() -> None:
    standard_english_quote_convention: QuoteConvention = QuoteConvention(
        "standard_english",
        [
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    standard_french_quote_convention: QuoteConvention = QuoteConvention(
        "standard_french",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u2039", "\u203a"),
        ],
    )

    western_european_quote_convention: QuoteConvention = QuoteConvention(
        "western_european",
        [
            SingleLevelQuoteConvention("\u00ab", "\u00bb"),
            SingleLevelQuoteConvention("\u201c", "\u201d"),
            SingleLevelQuoteConvention("\u2018", "\u2019"),
        ],
    )

    all_three_quote_convention_set = QuoteConventionSet(
        [
            standard_english_quote_convention,
            standard_french_quote_convention,
            western_european_quote_convention,
        ]
    )
    two_french_quote_convention_set = QuoteConventionSet(
        [western_european_quote_convention, standard_french_quote_convention]
    )

    multiple_english_quotes_tabulator = QuotationMarkTabulator()
    multiple_english_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 14, 15),
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 28, 29),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 42, 43),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(multiple_english_quotes_tabulator) == (
        standard_english_quote_convention,
        1.0,
    )

    multiple_western_european_quotes_tabulator = QuotationMarkTabulator()
    multiple_western_european_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u201c", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u201d", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 14, 15),
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 28, 29),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 42, 43),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(multiple_western_european_quotes_tabulator) == (
        western_european_quote_convention,
        1.0,
    )

    multiple_french_quotes_tabulator = QuotationMarkTabulator()
    multiple_french_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u2039", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u203a", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 14, 15),
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 28, 29),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 42, 43),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(multiple_french_quotes_tabulator) == (
        standard_french_quote_convention,
        1.0,
    )
    assert two_french_quote_convention_set.find_most_similar_convention(multiple_french_quotes_tabulator) == (
        standard_french_quote_convention,
        1.0,
    )

    noisy_multiple_english_quotes_tabulator = QuotationMarkTabulator()
    noisy_multiple_english_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u201c", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 14, 15),
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 28, 29),
            QuotationMarkMetadata("\u201d", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 42, 43),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(noisy_multiple_english_quotes_tabulator) == (
        standard_english_quote_convention,
        approx(0.9, rel=1e-9),
    )
    assert two_french_quote_convention_set.find_most_similar_convention(noisy_multiple_english_quotes_tabulator) == (
        western_european_quote_convention,
        approx(0.1, rel=1e-9),
    )

    noisy_multiple_french_quotes_tabulator = QuotationMarkTabulator()
    noisy_multiple_french_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u2039", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u203a", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u2039", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u2019", 2, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 14, 15),
            QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 28, 29),
            QuotationMarkMetadata("\u00bb", 1, QuotationMarkDirection.CLOSING, TextSegment.Builder().build(), 42, 43),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(noisy_multiple_french_quotes_tabulator) == (
        standard_french_quote_convention,
        approx(0.916666666666, rel=1e-9),
    )

    too_deep_english_quotes_tabulator = QuotationMarkTabulator()
    too_deep_english_quotes_tabulator.tabulate(
        [
            QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1),
            QuotationMarkMetadata("\u2018", 2, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 5, 6),
            QuotationMarkMetadata("\u201c", 3, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 13, 14),
            QuotationMarkMetadata("\u2018", 4, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 15, 16),
            QuotationMarkMetadata("\u201c", 5, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 17, 18),
        ]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(too_deep_english_quotes_tabulator) == (
        standard_english_quote_convention,
        approx(0.967741935483871, rel=1e-9),
    )

    # in case of ties, the earlier convention in the list should be returned
    unknown_quote_tabulator = QuotationMarkTabulator()
    unknown_quote_tabulator.tabulate(
        [QuotationMarkMetadata("\u201a", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1)]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(unknown_quote_tabulator) == (
        standard_english_quote_convention,
        0.0,
    )

    single_french_opening_quote_tabulator = QuotationMarkTabulator()
    single_french_opening_quote_tabulator.tabulate(
        [QuotationMarkMetadata("\u00ab", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1)]
    )
    assert all_three_quote_convention_set.find_most_similar_convention(single_french_opening_quote_tabulator) == (
        standard_french_quote_convention,
        1.0,
    )
    assert two_french_quote_convention_set.find_most_similar_convention(single_french_opening_quote_tabulator) == (
        western_european_quote_convention,
        1.0,
    )

    # Default values should be returned when the QuoteConventionSet is empty
    single_english_opening_quote_tabulator = QuotationMarkTabulator()
    single_english_opening_quote_tabulator.tabulate(
        [QuotationMarkMetadata("\u201c", 1, QuotationMarkDirection.OPENING, TextSegment.Builder().build(), 0, 1)]
    )
    empty_quote_convention_set = QuoteConventionSet([])
    assert empty_quote_convention_set.find_most_similar_convention(single_english_opening_quote_tabulator) == (
        None,
        float("-inf"),
    )
