from tests.corpora.dbl_bundle_test_environment import DblBundleTestEnvironment


def test_texts() -> None:
    with DblBundleTestEnvironment() as env:
        assert [t.id for t in env.corpus.texts] == ["MAT", "MRK"]


def test_get_text() -> None:
    with DblBundleTestEnvironment() as env:
        assert any(env.corpus.get_text("MAT").get_segments())
        assert not any(env.corpus.get_text("LUK").get_segments())
