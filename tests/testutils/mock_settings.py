from dynaconf.base import Settings


class MockSettings(Settings):
    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.update(settings)
