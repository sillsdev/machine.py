from abc import ABC, abstractmethod

from .usfm_update_block import UsfmUpdateBlock


class UsfmUpdateBlockHandler(ABC):
    @abstractmethod
    def process_block(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock: ...


class UsfmUpdateBlockHandlerError(Exception):
    def __init__(self, block: UsfmUpdateBlock, *args):
        self._block = block
        super().__init__(*args)

    @property
    def block(self):
        return self._block
