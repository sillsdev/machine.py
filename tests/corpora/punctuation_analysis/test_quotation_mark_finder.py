from machine.corpora.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuotationMarkFinder,
    QuotationMarkStringMatch,
    QuoteConventionSet,
    TextSegment,
)


def test_that_all_possible_quotation_marks_are_identified() -> None:
    quotation_mark_finder = QuotationMarkFinder(STANDARD_QUOTE_CONVENTIONS)
    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("\u201cSample Text\u201d").build()
    ) == [
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201cSample Text\u201d").build(), 0, 1),
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\u201cSample Text\u201d").build(), 12, 13),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("\"Sample Text'").build()
    ) == [
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\"Sample Text'").build(), 0, 1),
        QuotationMarkStringMatch(TextSegment.Builder().set_text("\"Sample Text'").build(), 12, 13),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("All \u201cthe \u2019English quotation\u2018 marks\u201d").build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201cthe \u2019English quotation\u2018 marks\u201d").build(), 4, 5
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201cthe \u2019English quotation\u2018 marks\u201d").build(), 9, 10
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201cthe \u2019English quotation\u2018 marks\u201d").build(), 27, 28
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201cthe \u2019English quotation\u2018 marks\u201d").build(), 34, 35
        ),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("All \u00abthe \u2039French quotation\u203a marks\u00bb").build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u00abthe \u2039French quotation\u203a marks\u00bb").build(), 4, 5
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u00abthe \u2039French quotation\u203a marks\u00bb").build(), 9, 10
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u00abthe \u2039French quotation\u203a marks\u00bb").build(), 26, 27
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u00abthe \u2039French quotation\u203a marks\u00bb").build(), 33, 34
        ),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("All \"the 'typewriter quotation marks").build()
    ) == [
        QuotationMarkStringMatch(TextSegment.Builder().set_text("All \"the 'typewriter quotation marks").build(), 4, 5),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \"the 'typewriter quotation marks").build(), 9, 10
        ),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            9,
            10,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            22,
            23,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            45,
            47,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            53,
            54,
        ),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            4,
            5,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            9,
            10,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            18,
            19,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            27,
            28,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            38,
            39,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("All \u00abthe \u201cWestern \u2018european\u2019 quotation\u201d marks\u00bb")
            .build(),
            45,
            46,
        ),
    ]

    assert quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder().set_text("All \u201ethe \u201aCentral European quotation\u2018 marks\u201c").build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201ethe \u201aCentral European quotation\u2018 marks\u201c").build(),
            4,
            5,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201ethe \u201aCentral European quotation\u2018 marks\u201c").build(),
            9,
            10,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201ethe \u201aCentral European quotation\u2018 marks\u201c").build(),
            36,
            37,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder().set_text("All \u201ethe \u201aCentral European quotation\u2018 marks\u201c").build(),
            43,
            44,
        ),
    ]


def test_that_it_uses_the_quote_convention_set() -> None:
    standard_english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("standard_english")
    assert standard_english_quote_convention is not None

    english_quotation_mark_finder = QuotationMarkFinder(QuoteConventionSet([standard_english_quote_convention]))
    assert (
        english_quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build()
        )
        == []
    )

    typewriter_english_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("typewriter_english")
    assert typewriter_english_quote_convention is not None

    typewriter_english_quotation_mark_finder = QuotationMarkFinder(
        QuoteConventionSet([typewriter_english_quote_convention])
    )
    assert typewriter_english_quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            53,
            54,
        )
    ]

    western_european_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("western_european")
    assert western_european_quote_convention is not None

    western_european_quotation_mark_finder = QuotationMarkFinder(
        QuoteConventionSet([western_european_quote_convention])
    )
    assert western_european_quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            22,
            23,
        )
    ]

    typewriter_western_european_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(
        "typewriter_western_european"
    )
    assert typewriter_western_european_quote_convention is not None

    typewriter_western_european_quotation_mark_finder = QuotationMarkFinder(
        QuoteConventionSet([typewriter_western_european_quote_convention])
    )
    assert typewriter_western_european_quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            45,
            47,
        ),
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            53,
            54,
        ),
    ]

    central_european_quote_convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name("central_european")
    assert central_european_quote_convention is not None

    central_european_quotation_mark_finder = QuotationMarkFinder(
        QuoteConventionSet([central_european_quote_convention])
    )
    assert central_european_quotation_mark_finder.find_all_potential_quotation_marks_in_text_segment(
        TextSegment.Builder()
        .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
        .build()
    ) == [
        QuotationMarkStringMatch(
            TextSegment.Builder()
            .set_text("This has \u201equotes from \u00bbdifferent conventions <<mixed 'together")
            .build(),
            9,
            10,
        )
    ]
