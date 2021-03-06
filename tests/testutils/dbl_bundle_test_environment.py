from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ContextManager

from machine.corpora import DblBundleTextCorpus

from .corpora_test_helpers import create_test_dbl_bundle


class DblBundleTestEnvironment(ContextManager["DblBundleTestEnvironment"]):
    def __init__(self) -> None:
        self._temp_dir = TemporaryDirectory()
        bundle_filename = create_test_dbl_bundle(Path(self._temp_dir.name))
        self._corpus = DblBundleTextCorpus(bundle_filename)

    @property
    def corpus(self) -> DblBundleTextCorpus:
        return self._corpus

    def __enter__(self) -> DblBundleTestEnvironment:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._temp_dir.cleanup()
