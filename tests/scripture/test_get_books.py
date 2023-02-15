from pytest import raises

from machine.scripture import get_books


def test_get_books() -> None:
    assert get_books("MAL") == {39}
    assert get_books("GEN,EXO") == {1, 2}
    assert get_books("GEN,EXO") == get_books(["GEN", "EXO"])
    assert get_books("OT") == {i for i in range(1, 40)}
    assert get_books("NT") == {i for i in range(40, 67)}
    whole_bible = {i for i in range(1, 67)}
    assert get_books("NT,OT") == whole_bible

    whole_bible.remove(2)  # EXO
    whole_bible.remove(41)  # MRK
    assert get_books("NT,OT,-MRK,-EXO") == whole_bible

    with raises(RuntimeError):
        # invalid name
        get_books("HELLO_WORLD")

    with raises(RuntimeError):
        # subtracting book from nothing
        get_books("-MRK")

    with raises(RuntimeError):
        # invalid subtracting name
        get_books("NT,OT,-HELLO_WORLD")

    with raises(RuntimeError):
        # subtracting book from wrong set
        get_books("OT,-MRK,NT")
