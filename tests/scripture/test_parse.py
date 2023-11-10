from pytest import raises

from machine.scripture import get_books, get_chapters


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


def test_get_chapters() -> None:
    assert get_chapters("MAL") == {39: []}
    assert get_chapters("GEN,EXO") == {1: [], 2: []}
    assert get_chapters("OT") == {i: [] for i in range(1, 40)}
    assert get_chapters("NT") == {i: [] for i in range(40, 67)}
    whole_bible = {i: [] for i in range(1, 67)}
    assert get_chapters("NT,OT") == whole_bible

    del whole_bible[2]  # EXO
    del whole_bible[41]  # MRK
    assert get_chapters("NT,OT,-MRK,-EXO") == whole_bible

    assert get_chapters("MAT;MRK") == {40: [], 41: []}
    assert get_chapters("MAT1,2,3") == {40: [1, 2, 3]}
    assert get_chapters("MAT400-500") == {}
    assert get_chapters("MAT1-4,12,9,100") == {40: [1, 2, 3, 4, 9, 12]}
    assert get_chapters("MAT-LUK") == {40: [], 41: [], 42: []}
    assert get_chapters("MAT1,2,3;MAT-LUK") == {40: [1, 2, 3], 41: [], 42: []}
    assert get_chapters("2JN-3JN;EXO1,8,3-5;GEN") == {1: [], 2: [1, 3, 4, 5, 8], 63: [], 64: []}

    assert get_chapters("NT;OT;-MRK;-EXO") == whole_bible
    test_bible = {i: [] for i in range(40, 67)}
    test_chapters_mat = [1, 2] + [i for i in range(6, 17)] + [i for i in range(18, 29)]
    test_bible[40] = test_chapters_mat
    test_chapters_rev = [i for i in range(1, 21)]
    test_bible[66] = test_chapters_rev
    assert get_chapters("NT;-MAT3-5,17;-REV21,22") == test_bible

    assert get_chapters("MAT40.usfm;MRK41.usfm1,2,3,4-6;LUK") == {40: [], 41: [1, 2, 3, 4, 5, 6], 42: []}

    with raises(RuntimeError):
        # invalid name
        get_chapters("HELLO_WORLD")

    with raises(RuntimeError):
        # subtracting book from nothing
        get_chapters("-MRK")

    with raises(RuntimeError):
        # invalid subtracting name
        get_chapters("NT;OT;-HELLO_WORLD")

    with raises(RuntimeError):
        # subtracting range
        get_chapters("OT;NT;-MAT-LUK")
