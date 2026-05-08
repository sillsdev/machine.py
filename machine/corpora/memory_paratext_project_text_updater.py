from typing import Dict

from .memory_paratext_project_file_handler import MemoryParatextProjectFileHandler
from .paratext_project_settings import ParatextProjectSettings
from .paratext_project_text_updater_base import ParatextProjectTextUpdaterBase


class MemoryParatextProjectTextUpdater(ParatextProjectTextUpdaterBase):
    def __init__(self, files: Dict[str, str], settings: ParatextProjectSettings) -> None:
        super().__init__(MemoryParatextProjectFileHandler(files), settings)
