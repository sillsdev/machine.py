from abc import ABC, abstractmethod

from .usfm_update_block import UsfmUpdateBlock


class UsfmUpdateBlockHandler(ABC):
    @abstractmethod
    def process_block(self, block: UsfmUpdateBlock) -> UsfmUpdateBlock: ...
