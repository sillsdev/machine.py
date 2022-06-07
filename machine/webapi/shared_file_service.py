from pathlib import Path

from clearml import StorageManager


class SharedFileService:
    def __init__(self, config: dict) -> None:
        self._config = config

    @property
    def data_dir(self) -> Path:
        return Path(self._config["data_dir"], self._config["build_id"])

    def download_file(self, filename: str) -> Path:
        uri = f"{self._build_uri}/{filename}"
        file_path = StorageManager.download_file(uri, str(self.data_dir))
        if file_path is None:
            raise RuntimeError(f"Failed to download file: {uri}")
        return Path(file_path)

    def upload_file(self, filename: str) -> None:
        file_path = self.data_dir / filename
        StorageManager.upload_file(str(file_path), f"{self._build_uri}/{filename}")

    @property
    def _build_uri(self) -> str:
        build_uri: str = self._config["build_uri_scheme"] + "://" + self._config["build_uri"]
        build_uri = build_uri.rstrip("/")
        return build_uri
