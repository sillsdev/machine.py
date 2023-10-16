from itertools import islice

from decoy import Decoy

from machine.annotations import Range
from machine.tokenization import WHITESPACE_TOKENIZER
from machine.translation import (
    MAX_SEGMENT_LENGTH,
    InteractiveTranslationEngine,
    InteractiveTranslator,
    InteractiveTranslatorFactory,
    TranslationSources,
    WordAlignmentMatrix,
    WordGraph,
    WordGraphArc,
)

_SOURCE_SEGMENT = "En el principio la Palabra ya existía ."


def test_get_current_results_empty_prefix(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()

    result = next(translator.get_current_results())
    assert result.translation == "In the beginning the Word already existía ."


def test_get_current_results_append_complete_word(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In ")

    result = next(translator.get_current_results())
    assert result.translation == "In the beginning the Word already existía ."


def test_get_current_results_append_partial_word(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In ")
    translator.append_to_prefix("t")

    result = next(translator.get_current_results())
    assert result.translation == "In the beginning the Word already existía ."


def test_get_current_results_remove_word(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In the beginning ")
    translator.set_prefix("In the ")

    result = next(translator.get_current_results())
    assert result.translation == "In the beginning the Word already existía ."


def test_get_current_results_remove_all_words(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In the beginning ")
    translator.set_prefix("")

    result = next(translator.get_current_results())
    assert result.translation == "In the beginning the Word already existía ."


def test_is_source_segment_valid_valid(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()

    assert translator.is_segment_valid


def test_is_source_segment_valid_invalid(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    source_segment = ""
    for _ in range(MAX_SEGMENT_LENGTH):
        source_segment += "word "
    source_segment += "."
    decoy.when(env.engine.get_word_graph(source_segment)).then_return(
        WordGraph(WHITESPACE_TOKENIZER.tokenize(source_segment))
    )
    translator = env.create_translator(source_segment)

    assert not translator.is_segment_valid


def test_approve_aligned_only(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In the beginning ")
    translator.approve(aligned_only=True)

    decoy.verify(env.engine.train_segment("En el principio", "In the beginning", sentence_start=True), times=1)

    translator.append_to_prefix("the Word already existed .")
    translator.approve(aligned_only=True)

    decoy.verify(
        env.engine.train_segment(
            "En el principio la Palabra ya existía .",
            "In the beginning the Word already existed .",
            sentence_start=True,
        ),
        times=1,
    )


def test_approve_whole_source_segment(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    translator = env.create_translator()
    translator.append_to_prefix("In the beginning ")
    translator.approve(aligned_only=False)

    decoy.verify(
        env.engine.train_segment("En el principio la Palabra ya existía .", "In the beginning", sentence_start=True),
        times=1,
    )


def test_get_current_results_multiple_suggestions_empty_prefix(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.use_simple_word_graph()
    translator = env.create_translator()

    results = list(islice(translator.get_current_results(), 2))
    assert results[0].translation == "In the beginning the Word already existía ."
    assert results[1].translation == "In the start the Word already existía ."


def test_get_current_results_multiple_suggestions_nonempty_prefix(decoy: Decoy) -> None:
    env = _TestEnvironment(decoy)
    env.use_simple_word_graph()
    translator = env.create_translator()
    translator.append_to_prefix("In the ")

    results = list(islice(translator.get_current_results(), 2))
    assert results[0].translation == "In the beginning the Word already existía ."
    assert results[1].translation == "In the start the Word already existía ."

    translator.append_to_prefix("beginning")

    results = list(islice(translator.get_current_results(), 2))
    assert results[0].translation == "In the beginning the Word already existía ."
    assert results[1].translation == "In the beginning his Word already existía ."


class _TestEnvironment:
    def __init__(self, decoy: Decoy) -> None:
        self._decoy = decoy
        self.engine = decoy.mock(cls=InteractiveTranslationEngine)

        word_graph = WordGraph(
            source_tokens=WHITESPACE_TOKENIZER.tokenize(_SOURCE_SEGMENT),
            arcs=[
                WordGraphArc(
                    0,
                    1,
                    -22.4162,
                    ["now", "it"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 1), (1, 0)]),
                    Range.create(0, 2),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.00006755903, 0.0116618536],
                ),
                WordGraphArc(
                    0,
                    2,
                    -23.5761,
                    ["In", "your"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(0, 2),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.355293363, 0.0000941652761],
                ),
                WordGraphArc(
                    0,
                    3,
                    -11.1167,
                    ["In", "the"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(0, 2),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.355293363, 0.5004668],
                ),
                WordGraphArc(
                    0,
                    4,
                    -13.7804,
                    ["In"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(0, 1),
                    [TranslationSources.SMT],
                    [0.355293363],
                ),
                WordGraphArc(
                    3,
                    5,
                    -12.9695,
                    ["beginning"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(2, 3),
                    [TranslationSources.SMT],
                    [0.348795831],
                ),
                WordGraphArc(
                    4,
                    5,
                    -7.68319,
                    ["the", "beginning"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(1, 3),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.5004668, 0.348795831],
                ),
                WordGraphArc(
                    4,
                    3,
                    -14.4373,
                    ["the"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(1, 2),
                    [TranslationSources.SMT],
                    [0.5004668],
                ),
                WordGraphArc(
                    5,
                    6,
                    -19.3042,
                    ["his", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.00347203249, 0.477621228],
                ),
                WordGraphArc(
                    5,
                    7,
                    -8.49148,
                    ["the", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.346071422, 0.477621228],
                ),
                WordGraphArc(
                    1,
                    8,
                    -15.2926,
                    ["beginning"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(2, 3),
                    [TranslationSources.SMT],
                    [0.348795831],
                ),
                WordGraphArc(
                    2,
                    9,
                    -15.2926,
                    ["beginning"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(2, 3),
                    [TranslationSources.SMT],
                    [0.348795831],
                ),
                WordGraphArc(
                    7,
                    10,
                    -14.3453,
                    ["already"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(5, 6),
                    [TranslationSources.SMT],
                    [0.2259867],
                ),
                WordGraphArc(
                    8,
                    6,
                    -19.3042,
                    ["his", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.00347203249, 0.477621228],
                ),
                WordGraphArc(
                    8,
                    7,
                    -8.49148,
                    ["the", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.346071422, 0.477621228],
                ),
                WordGraphArc(
                    9,
                    6,
                    -19.3042,
                    ["his", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.00347203249, 0.477621228],
                ),
                WordGraphArc(
                    9,
                    7,
                    -8.49148,
                    ["the", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.346071422, 0.477621228],
                ),
                WordGraphArc(
                    6,
                    10,
                    -14.0526,
                    ["already"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(5, 6),
                    [TranslationSources.SMT],
                    [0.2259867],
                ),
                WordGraphArc(
                    10,
                    11,
                    51.1117,
                    ["existía"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(6, 7),
                    [TranslationSources.NONE],
                    [0.0],
                ),
                WordGraphArc(
                    11,
                    12,
                    -29.0049,
                    ["you", "."],
                    WordAlignmentMatrix.from_word_pairs(1, 2, [(0, 1)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.005803475, 0.317073762],
                ),
                WordGraphArc(
                    11,
                    13,
                    -27.7143,
                    ["to"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.038961038],
                ),
                WordGraphArc(
                    11,
                    14,
                    -30.0868,
                    [".", "‘"],
                    WordAlignmentMatrix.from_word_pairs(1, 2, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.317073762, 0.06190489],
                ),
                WordGraphArc(
                    11,
                    15,
                    -30.1586,
                    [".", "he"],
                    WordAlignmentMatrix.from_word_pairs(1, 2, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.317073762, 0.06702433],
                ),
                WordGraphArc(
                    11,
                    16,
                    -28.2444,
                    [".", "the"],
                    WordAlignmentMatrix.from_word_pairs(1, 2, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.317073762, 0.115540564],
                ),
                WordGraphArc(
                    11,
                    17,
                    -23.8056,
                    ["and"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.08047272],
                ),
                WordGraphArc(
                    11,
                    18,
                    -23.5842,
                    ["the"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.09361572],
                ),
                WordGraphArc(
                    11,
                    19,
                    -18.8988,
                    [","],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.1428188],
                ),
                WordGraphArc(
                    11,
                    20,
                    -11.9218,
                    [".", "’"],
                    WordAlignmentMatrix.from_word_pairs(1, 2, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.317073762, 0.018057242],
                ),
                WordGraphArc(
                    11,
                    21,
                    -3.51852,
                    ["."],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.317073762],
                ),
            ],
            final_states=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            initial_state_score=-191.0998,
        )

        decoy.when(self.engine.get_word_graph(_SOURCE_SEGMENT)).then_return(word_graph)

        self._factory = InteractiveTranslatorFactory(self.engine)

    def use_simple_word_graph(self) -> None:
        word_graph = WordGraph(
            source_tokens=WHITESPACE_TOKENIZER.tokenize(_SOURCE_SEGMENT),
            arcs=[
                WordGraphArc(
                    0,
                    1,
                    -10,
                    ["In", "the", "beginning"],
                    WordAlignmentMatrix.from_word_pairs(3, 3, [(0, 0), (1, 1), (2, 2)]),
                    Range.create(0, 3),
                    [TranslationSources.SMT, TranslationSources.SMT, TranslationSources.SMT],
                    [0.5, 0.5, 0.5],
                ),
                WordGraphArc(
                    0,
                    1,
                    -11,
                    ["In", "the", "start"],
                    WordAlignmentMatrix.from_word_pairs(3, 3, [(0, 0), (1, 1), (2, 2)]),
                    Range.create(0, 3),
                    [TranslationSources.SMT, TranslationSources.SMT, TranslationSources.SMT],
                    [0.5, 0.5, 0.4],
                ),
                WordGraphArc(
                    1,
                    2,
                    -10,
                    ["the", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.5, 0.5],
                ),
                WordGraphArc(
                    1,
                    2,
                    -11,
                    ["his", "Word"],
                    WordAlignmentMatrix.from_word_pairs(2, 2, [(0, 0), (1, 1)]),
                    Range.create(3, 5),
                    [TranslationSources.SMT, TranslationSources.SMT],
                    [0.4, 0.5],
                ),
                WordGraphArc(
                    2,
                    3,
                    -10,
                    ["already"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(5, 6),
                    [TranslationSources.SMT],
                    [0.5],
                ),
                WordGraphArc(
                    3,
                    4,
                    50,
                    ["existía"],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(6, 7),
                    [TranslationSources.NONE],
                    [0.0],
                ),
                WordGraphArc(
                    4,
                    5,
                    -10,
                    ["."],
                    WordAlignmentMatrix.from_word_pairs(1, 1, [(0, 0)]),
                    Range.create(7, 8),
                    [TranslationSources.SMT],
                    [0.5],
                ),
            ],
            final_states=[5],
        )
        self._decoy.when(self.engine.get_word_graph(_SOURCE_SEGMENT)).then_return(word_graph)

    def create_translator(self, segment: str = _SOURCE_SEGMENT) -> InteractiveTranslator:
        return self._factory.create(segment)
