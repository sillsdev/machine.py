from abc import ABC, abstractmethod
from typing import BinaryIO, List, Optional, Tuple, Union

from machine.corpora.paratext_project_settings import ParatextProjectSettings
from machine.corpora.paratext_project_settings_parser_base import ParatextProjectSettingsParserBase
from machine.corpora.scripture_ref import ScriptureRef
from machine.corpora.update_usfm_parser_handler import UpdateUsfmParserHandler
from machine.corpora.usfm_parser import parse_usfm

from ..utils.typeshed import StrPath


class ParatextProjectTextUpdaterBase(ABC):
    def __init__(self, settings: Union[ParatextProjectSettings, ParatextProjectSettingsParserBase]) -> None:
        if isinstance(settings, ParatextProjectSettingsParserBase):
            self._settings = settings.parse()
        else:
            self._settings = settings

    def update_usfm(
        self,
        book_id: str,
        rows: Optional[List[Tuple[List[ScriptureRef], str]]] = None,
        full_name: Optional[str] = None,
        strip_all_text: bool = False,
        prefer_existing_text: bool = True,
    ) -> Optional[str]:
        file_name: str = self._settings.get_book_file_name(book_id)
        if not self.exists(file_name):
            return None
        with self.open(file_name) as sfm_file:
            usfm: str = sfm_file.read().decode(self._settings.encoding)
        handler = UpdateUsfmParserHandler(
            rows, None if full_name is None else f"- {full_name}", strip_all_text, prefer_existing_text
        )
        try:
            parse_usfm(usfm, handler, self._settings.stylesheet, self._settings.versification)
            return handler.get_usfm(self._settings.stylesheet)
        except Exception as e:
            error_message = (
                f"An error occurred while parsing the usfm for '{book_id}'"
                f"{f' in project {self._settings.name}' if self._settings.name else ''}"
                f". Error: '{e}'"
            )
            raise RuntimeError(error_message) from e

    @abstractmethod
    def exists(self, file_name: StrPath) -> bool: ...

    @abstractmethod
    def open(self, file_name: StrPath) -> BinaryIO: ...
