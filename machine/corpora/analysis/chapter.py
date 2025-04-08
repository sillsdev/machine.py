from .verse import Verse


class Chapter:
    def __init__(self, verses: list[Verse]):
        self.verses = verses

    def get_verses(self) -> list[Verse]:
        return self.verses
