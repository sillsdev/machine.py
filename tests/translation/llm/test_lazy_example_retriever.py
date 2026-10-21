import threading
from concurrent.futures import ThreadPoolExecutor

from testutils.mock_example_retriever import MockExampleRetriever

from machine.translation.llm.example import Example


class _BlockingFitExampleRetriever(MockExampleRetriever):
    def __init__(self, examples):
        super().__init__(examples)
        self.fit_calls = 0
        self.fit_started = threading.Event()
        self.release_fit = threading.Event()
        self.second_fit_started = threading.Event()
        self._fit_counter_lock = threading.Lock()

    def _fit(self) -> None:
        with self._fit_counter_lock:
            self.fit_calls += 1
            if self.fit_calls == 2:
                self.second_fit_started.set()
        self.fit_started.set()
        self.release_fit.wait(timeout=5)


def test_lazy_example_retriever_fits_once_when_threads_rank_together():
    retriever = _BlockingFitExampleRetriever([Example("a", "1"), Example("b", "2"), Example("c", "3")])
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(retriever.rank, "a", 1)
        assert retriever.fit_started.wait(timeout=1)
        second = executor.submit(retriever.rank, "b", 1)
        try:
            assert not retriever.second_fit_started.wait(timeout=0.2)
        finally:
            retriever.release_fit.set()
        assert first.result(timeout=1)
        assert second.result(timeout=1)

    assert retriever.fit_calls == 1


def test_lazy_example_retriever_k_greater_than_n():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 3)
    assert retrieved_examples == examples


def test_lazy_example_retriever_no_such_exclude():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 2, Example("c", "3"))
    assert retrieved_examples == examples


def test_lazy_example_retriever_exclude_index():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 2, 0)
    assert retrieved_examples == examples[1:]


def test_lazy_example_retriever_exclude_example():
    examples = [Example("a", "1"), Example("b", "2")]
    retriever = MockExampleRetriever(examples)
    retrieved_examples = retriever.rank("query", 2, Example("a", "1"))
    assert retrieved_examples == examples[1:]
