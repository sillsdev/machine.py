from abc import abstractmethod

from .stream_container import StreamContainer
from .text_base import TextBase


class StreamTextBase(TextBase):
    @abstractmethod
    def _create_stream_container(self) -> StreamContainer:
        ...
