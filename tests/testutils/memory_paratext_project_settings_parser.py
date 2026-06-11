from typing import Optional

from machine.corpora import ParatextProjectSettings, ParatextProjectSettingsParserBase

from .memory_paratext_project_file_handler import MemoryParatextProjectFileHandler


class MemoryParatextProjectSettingsParser(ParatextProjectSettingsParserBase):
    def __init__(
        self,
        files: Optional[dict[str, str]] = None,
        parent_settings: Optional[ParatextProjectSettings] = None,
    ):
        super().__init__(
            MemoryParatextProjectFileHandler(files),
            parent_settings,
        )
