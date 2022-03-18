from testutils.dbl_bundle_test_environment import DblBundleTestEnvironment


def test_texts() -> None:
    with DblBundleTestEnvironment() as env:
        assert [t.id for t in env.corpus.texts] == ["MAT", "MRK"]


def test_get_text() -> None:
    with DblBundleTestEnvironment() as env:
        mat = env.corpus.get_text("MAT")
        assert mat is not None
        assert any(mat.get_rows())

        luk = env.corpus.get_text("LUK")
        assert luk is None
