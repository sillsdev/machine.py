from .text_segment import TextSegment


class Verse:
    def __init__(self, text_segments: list[TextSegment]):
        self.text_segments = text_segments
        self._index_text_segments()

    def _index_text_segments(self) -> None:
        for index, text_segment in enumerate(self.text_segments):
            text_segment.set_index_in_verse(index)
            text_segment.set_num_segments_in_verse(len(self.text_segments))

    def get_text_segments(self) -> list[TextSegment]:
        return self.text_segments
