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
    assert get_books("MAT-JHN") == {i for i in range(40, 44)}
    assert get_books("MAT-REV") == {i for i in range(40, 67)}
    assert get_books("MAT-JHN;ACT") == {i for i in range(40, 45)}
    assert get_books("MAT-JHN;ACT;-JHN-ACT,REV") == {40, 41, 42, 66}

    with raises(ValueError):
        # invalid name
        get_books("HELLO_WORLD")

    with raises(ValueError):
        # subtracting book from nothing
        get_books("-MRK")

    with raises(ValueError):
        # invalid subtracting name
        get_books("NT,OT,-HELLO_WORLD")

    with raises(ValueError):
        # subtracting book from wrong set
        get_books("OT,-MRK,NT")

    with raises(ValueError):
        # invalid range book
        get_books("MAT-ABC")

    with raises(ValueError):
        # subtract invalid range book
        get_books("NT;-ABC-LUK")

    with raises(ValueError):
        # invalid range order
        get_books("MAT-GEN")


def test_get_chapters() -> None:
    assert get_chapters([]) == {}
    assert get_chapters("MAL") == {39: []}
    assert get_chapters("PS2") == {84: []}
    assert get_chapters("GEN,EXO") == {1: [], 2: []}
    assert get_chapters("1JN,2JN") == {62: [], 63: []}
    assert get_chapters("OT") == {i: [] for i in range(1, 40)}
    assert get_chapters("NT") == {i: [] for i in range(40, 67)}
    whole_bible = {i: [] for i in range(1, 67)}
    assert get_chapters("NT,OT") == whole_bible

    assert get_chapters("MAT;MRK") == {40: [], 41: []}
    assert get_chapters("MAT; MRK") == {40: [], 41: []}
    assert get_chapters("MAT1,2,3") == {40: [1, 2, 3]}
    assert get_chapters("MAT 1, 2, 3") == {40: [1, 2, 3]}
    assert get_chapters("MAT1-4,12,9") == {40: [1, 2, 3, 4, 9, 12]}
    assert get_chapters("MAT-LUK") == {40: [], 41: [], 42: []}
    assert get_chapters("MAT1,2,3;MAT-LUK") == {40: [], 41: [], 42: []}
    assert get_chapters("2JN-3JN;EXO1,8,3-5;GEN") == {1: [], 2: [1, 3, 4, 5, 8], 63: [], 64: []}
    assert get_chapters("1JN 1;1JN 2;1JN 3-5") == {62: []}
    assert get_chapters("MAT-ROM;-ACT4-28") == {40: [], 41: [], 42: [], 43: [], 44: [1, 2, 3], 45: []}
    assert get_chapters("2JN;-2JN 1") == {}

    del whole_bible[2]  # EXO
    del whole_bible[41]  # MRK
    assert get_chapters("NT;OT;-MRK;-EXO") == whole_bible
    test_bible = {i: [] for i in range(40, 67)}
    test_chapters_mat = [1, 2] + [i for i in range(6, 17)] + [i for i in range(18, 29)]
    test_bible[40] = test_chapters_mat
    test_chapters_rev = [i for i in range(1, 21)]
    test_bible[66] = test_chapters_rev
    assert get_chapters("NT;-MAT3-5,17;-REV21,22") == test_bible
    assert get_chapters("MAT-JHN;-MAT-LUK") == {43: []}
    assert get_chapters("") == {}

    with raises(ValueError):
        # wrong order of chapters
        print(get_chapters("MAT3-1"))

    with raises(ValueError):
        # wrong order of chapters in subtraction
        print(get_chapters("MRK;-MRK10-3"))

    with raises(ValueError):
        # chapter non-range
        print(get_chapters("MAT1-1"))

    with raises(ValueError):
        # chapter non-range in subtraction
        print(get_chapters("MRK;-MRK3-3"))

    with raises(ValueError):
        # wrong order of books
        print(get_chapters("MRK-MAT"))

    with raises(ValueError):
        # wrong order of books in subtraction
        print(get_chapters("-MRK-MAT"))

    with raises(ValueError):
        # book non-range
        print(get_chapters("MAT-MAT"))

    with raises(ValueError):
        # book non-range in subtraction
        print(get_chapters("-MRK-MRK"))

    with raises(ValueError):
        # chapter 0
        print(get_chapters("MAT0-10"))

    with raises(ValueError):
        # invalid book/length of book name
        get_chapters("MAT-FLUM")

    with raises(ValueError):
        # invalid book/length of book name in subtraction
        get_chapters("-MAT-FLUM")

    with raises(ValueError):
        # invalid name
        get_chapters("ABC")

    with raises(ValueError):
        # invalid range book
        get_chapters("MAT-ABC")
    with raises(ValueError):
        # subtract invalid range book
        get_chapters("NT;-ABC-LUK")

    with raises(ValueError):
        # invalid chapter
        get_chapters("MAT 500")

    with raises(ValueError):
        # invalid range number
        get_chapters("MAT 1-500")
    with raises(ValueError):
        # subtract invalid range number
        get_chapters("MAT;-MAT 300-500")

    # subtracting from nothing
    with raises(ValueError):
        get_chapters("-MRK")
    with raises(ValueError):
        get_chapters("-MRK 1")
    with raises(ValueError):
        get_chapters("MRK 2-5;-MRK 1-4")
    with raises(ValueError):
        get_chapters("MRK 2-5;-MRK 6")
    with raises(ValueError):
        get_chapters("MRK 1-3;-MRK")
    with raises(ValueError):
        get_chapters("OT;-MRK-LUK")

    # invalid subtracting name
    with raises(ValueError):
        get_chapters("NT;OT;-ABC")
    with raises(ValueError):
        get_chapters("MAT;-ABC 1")

    # mixing old (comma-separated) and new syntax
    with raises(ValueError):
        get_chapters("NT,OT,-MRK,-EXO")
    with raises(ValueError):
        get_chapters("OT,MAT1")
    with raises(ValueError):
        get_chapters("OT,MAT-LUK")
