import pytest

from machine.translation.llm import ContextVerse, PostEditSegment


def test_init_context_verse_offset_zero() -> None:
    with pytest.raises(ValueError, match="offset 0"):
        PostEditSegment("the dog", "la perro", (ContextVerse(0, "the dog", "el perro"),))


def test_init_duplicate_context_verse_offsets() -> None:
    with pytest.raises(ValueError, match="distinct offsets"):
        PostEditSegment("the dog", "la perro", (ContextVerse(1, "after", "después"), ContextVerse(1, "again")))
