from typing import Generator

from .corpora_helpers import gen
from .scripture_text import ScriptureText
from .text_segment import TextSegment


class NullScriptureText(ScriptureText):
    def _get_segments_in_doc_order(self, include_text: bool) -> Generator[TextSegment, None, None]:
        return gen()
