from .quote_convention import QuoteConvention, SingleLevelQuoteConvention
from .quote_convention_set import QuoteConventionSet

STANDARD_QUOTE_CONVENTIONS: QuoteConventionSet = QuoteConventionSet(
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
            "typewriter_english",
            [
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "british_english",
            [
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
            ],
        ),
        QuoteConvention(
            "british_typewriter_english",
            [
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
            ],
        ),
        QuoteConvention(
            "hybrid_typewriter_english",
            [
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
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
            "french_variant",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2039", "\u203a"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
            ],
        ),
        QuoteConvention(
            "western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
            ],
        ),
        QuoteConvention(
            "british_inspired_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
            ],
        ),
        QuoteConvention(
            "typewriter_western_european",
            [
                SingleLevelQuoteConvention("<<", ">>"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "typewriter_western_european_variant",
            [
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("<", ">"),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "hybrid_typewriter_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "hybrid_british_typewriter_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
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
            "central_european_guillemets",
            [
                SingleLevelQuoteConvention("\u00bb", "\u00ab"),
                SingleLevelQuoteConvention("\u203a", "\u2039"),
                SingleLevelQuoteConvention("\u00bb", "\u00ab"),
                SingleLevelQuoteConvention("\u203a", "\u2039"),
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
        QuoteConvention(
            "standard_finnish",
            [
                SingleLevelQuoteConvention("\u00bb", "\u00bb"),
                SingleLevelQuoteConvention("\u2019", "\u2019"),
            ],
        ),
        QuoteConvention(
            "eastern_european",
            [
                SingleLevelQuoteConvention("\u201e", "\u201d"),
                SingleLevelQuoteConvention("\u201a", "\u2019"),
                SingleLevelQuoteConvention("\u201e", "\u201d"),
                SingleLevelQuoteConvention("\u201a", "\u2019"),
            ],
        ),
        QuoteConvention(
            "standard_russian",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
            ],
        ),
        QuoteConvention(
            "standard_arabic",
            [
                SingleLevelQuoteConvention("\u201d", "\u201c"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
                SingleLevelQuoteConvention("\u201d", "\u201c"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
            ],
        ),
        QuoteConvention(
            "non-standard_arabic",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
            ],
        ),
    ]
)
