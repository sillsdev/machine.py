from machine.translation.llm import Example
from machine.translation.llm.coverage_example_retriever import CoverageExampleRetriever


def test_coverage_retriever_prefers_an_uncovered_clause_to_a_near_duplicate():
    sources = [
        "the shepherd counted his sheep in the green field",
        "the shepherd counted his sheep in the green field again",
        "the shepherd counted all his sheep in the green field",
        "fishermen mended nets",
        "a woman drew water from the well",
    ]
    query = "the shepherd counted his sheep in the green field while fishermen mended nets"

    retriever = CoverageExampleRetriever([Example(source, "") for source in sources])

    assert [example.source for example in retriever.rank(query, 2)] == [sources[0], sources[3]]


def test_coverage_retriever_keeps_the_closest_example():
    sources = ["the shepherd counted his sheep in the green field", "fishermen mended nets"]
    query = "the shepherd counted his sheep in the green field while fishermen mended nets"

    retriever = CoverageExampleRetriever([Example(source, "") for source in sources])

    assert [example.source for example in retriever.rank(query, 1)] == [sources[0]]


def test_coverage_retriever_selects_an_exact_duplicate_pair_only_once():
    examples = [Example("apple pie", "tarte"), Example("apple pie", "tarte"), Example("banana split", "banane")]
    retriever = CoverageExampleRetriever(examples)

    selected = retriever.rank("apple pie", 2)

    assert examples[2] in selected
    assert len([example for example in selected if example == examples[0]]) == 1


def test_coverage_retriever_keeps_repeated_sources_with_different_targets():
    examples = [Example("apple pie", "tarte"), Example("apple pie", "tourte"), Example("banana split", "banane")]

    assert set(CoverageExampleRetriever(examples).rank("apple pie", 2)) == set(examples[:2])


def test_coverage_retriever_excludes_copies_of_the_excluded_example():
    examples = [Example("apple pie", "tarte"), Example("apple pie", "tarte"), Example("banana split", "banane")]

    assert CoverageExampleRetriever(examples).rank("apple pie", 2, exclude=0) == [examples[2]]


def test_coverage_retriever_finds_nothing_when_all_remaining_examples_are_copies():
    examples = [Example("apple pie", "tarte"), Example("apple pie", "tarte")]

    assert CoverageExampleRetriever(examples).rank("apple pie", 1, exclude=0) == []


def test_coverage_retriever_matches_a_script_without_spaces():
    sources = ["他们在家里吃饭", "我们去学校", "明天下雨"]

    retriever = CoverageExampleRetriever([Example(source, "") for source in sources])

    assert retriever.rank("我们明天去学校吗", 1)[0].source == sources[1]


def test_coverage_retriever_fills_k_when_the_query_has_no_known_ngrams():
    retriever = CoverageExampleRetriever(
        [Example("apple pie", ""), Example("banana split", ""), Example("cherry tart", "")]
    )

    assert len(set(retriever.rank("zzzz", 2))) == 2


def test_coverage_retriever_handles_a_corpus_with_no_text():
    retriever = CoverageExampleRetriever([Example("", ""), Example("  ", "")])

    assert retriever.rank("anything", 1) == []
