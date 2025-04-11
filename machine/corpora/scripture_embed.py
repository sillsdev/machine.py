from typing import Optional

EMBED_PART_START_CHAR_STYLES = ("f", "x", "z")
EMBED_STYLES = ("f", "fe", "fig", "fm", "x")


def is_note_text(marker: Optional[str]) -> bool:
    return marker == "ft"


def is_embed_part_style(marker: Optional[str]) -> bool:
    return marker is not None and marker.startswith(EMBED_PART_START_CHAR_STYLES)


def is_embed_style(marker: Optional[str]) -> bool:
    return marker is not None and marker.strip("*") in EMBED_STYLES
