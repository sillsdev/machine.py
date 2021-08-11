import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence, overload


class TokenProcessor(ABC):
    @abstractmethod
    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        ...


class LowercaseTokenProcessor(TokenProcessor):
    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        return [t.lower() for t in tokens]


class EscapeSpacesTokenProcessor(TokenProcessor):
    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        return [("<space>" if len(t) > 0 and t.isspace() else t) for t in tokens]


class UnescapeSpacesTokenProcessor(TokenProcessor):
    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        return [(" " if t == "<space>" else t) for t in tokens]


class NoOpTokenProcessor(TokenProcessor):
    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        return tokens


LOWERCASE = LowercaseTokenProcessor()
ESCAPE_SPACES = EscapeSpacesTokenProcessor()
UNESCAPE_SPACES = UnescapeSpacesTokenProcessor()
NO_OP = NoOpTokenProcessor()


class NormalizeTokenProcessor(TokenProcessor):
    def __init__(self, normalization_form: str) -> None:
        self._normalization_form = normalization_form

    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        return [unicodedata.normalize(self._normalization_form, t) for t in tokens]


NFC_NORMALIZE = NormalizeTokenProcessor("NFC")
NFD_NORMALIZE = NormalizeTokenProcessor("NFD")
NFKC_NORMALIZE = NormalizeTokenProcessor("NFKC")
NFKD_NORMALIZE = NormalizeTokenProcessor("NFKD")


class PipelineTokenProcessor(TokenProcessor):
    @overload
    def __init__(self, *processors: TokenProcessor) -> None:
        ...

    @overload
    def __init__(self, processors: Iterable[TokenProcessor]) -> None:
        ...

    def __init__(self, *args: Any) -> None:
        if len(args) == 0:
            self._processors = []
        elif isinstance(args[0], TokenProcessor):
            self._processors = args
        else:
            self._processors = list(args[0])

    def process(self, tokens: Sequence[str]) -> Sequence[str]:
        for processor in self._processors:
            if processor is None:
                processor = NO_OP
            tokens = processor.process(tokens)
        return tokens


@overload
def pipeline(*processors: TokenProcessor) -> PipelineTokenProcessor:
    ...


@overload
def pipeline(processors: Iterable[TokenProcessor]) -> PipelineTokenProcessor:
    ...


def pipeline(*args: Any) -> PipelineTokenProcessor:
    return PipelineTokenProcessor(args)
