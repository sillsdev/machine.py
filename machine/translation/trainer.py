from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import Callable, ContextManager, Dict, Optional, Type

from ..utils.progress_status import ProgressStatus


class TrainStats:
    def __init__(self) -> None:
        self.train_corpus_size: int = 0
        self._metrics: Dict[str, float] = {}

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics


class Trainer(ContextManager["Trainer"]):
    @abstractmethod
    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @property
    @abstractmethod
    def stats(self) -> TrainStats:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> Trainer:
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        self.close()
