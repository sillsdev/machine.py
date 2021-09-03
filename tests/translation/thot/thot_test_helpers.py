from pathlib import Path

from machine.corpora import DictionaryTextCorpus, MemoryText, ParallelTextCorpus, TextSegment, TextSegmentRef

TOY_CORPUS_HMM_PATH = Path(__file__).parent / "data" / "toy_corpus_hmm"
TOY_CORPUS_HMM_CONFIG_FILENAME = TOY_CORPUS_HMM_PATH / "smt.cfg"

TOY_CORPUS_FAST_ALIGN_PATH = Path(__file__).parent / "data" / "toy_corpus_fa"
TOY_CORPUS_FAST_ALIGN_CONFIG_FILENAME = TOY_CORPUS_FAST_ALIGN_PATH / "smt.cfg"


def create_test_parallel_corpus() -> ParallelTextCorpus:
    src_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _segment(1, "isthay isyay ayay esttay-N ."),
                _segment(2, "ouyay ouldshay esttay-V oftenyay ."),
                _segment(3, "isyay isthay orkingway ?"),
                _segment(4, "isthay ouldshay orkway-V ."),
                _segment(5, "ityay isyay orkingway ."),
                _segment(6, "orkway-N ancay ebay ardhay !"),
                _segment(7, "ayay esttay-N ancay ebay ardhay ."),
                _segment(8, "isthay isyay ayay ordway !"),
            ],
        )
    )

    trg_corpus = DictionaryTextCorpus(
        MemoryText(
            "text1",
            [
                _segment(1, "this is a test N ."),
                _segment(2, "you should test V often ."),
                _segment(3, "is this working ?"),
                _segment(4, "this should work V ."),
                _segment(5, "it is working ."),
                _segment(6, "work N can be hard !"),
                _segment(7, "a test N can be hard ."),
                _segment(8, "this is a word !"),
            ],
        )
    )

    return ParallelTextCorpus(src_corpus, trg_corpus)


def _segment(ref: int, segment: str) -> TextSegment:
    return TextSegment(
        "text1",
        TextSegmentRef(ref),
        segment.split(),
        is_sentence_start=True,
        is_in_range=False,
        is_range_start=False,
        is_empty=False,
    )
