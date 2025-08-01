from .text_segment import TextSegment


class Verse:
    def __init__(self, text_segments: list[TextSegment]):
        self._text_segments = text_segments
        self._index_text_segments()

    def _index_text_segments(self) -> None:
        for index, text_segment in enumerate(self._text_segments):
            text_segment.index_in_verse = index
            text_segment.num_segments_in_verse = len(self._text_segments)

    @property
    def text_segments(self) -> list[TextSegment]:
        return self._text_segments
