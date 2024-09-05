from typing import Dict, List, Optional, Tuple

from machine.corpora import ParatextProjectSettings, ParatextProjectTermsParserBase, UsfmStylesheet
from machine.corpora.paratext_project_terms_parser_base import _get_glosses, _strip_parens
from machine.scripture import ORIGINAL_VERSIFICATION, Versification
from tests.corpora.memory_paratext_project_terms_parser import MemoryParatextProjectTermsParser


def test_get_key_terms_from_terms_renderings() -> None:
    env = _TestEnvironment(
        files={
            "ProjectBiblicalTerms.xml": r"""
<BiblicalTermsList>
  <Term Id="אֲחַשְׁוֵרוֹשׁ">
  <Category>PN</Category>
    <Gloss>Ahasuerus</Gloss>
  </Term>
</BiblicalTermsList>
""",
            "TermRenderings.xml": r"""
<TermRenderingsList>
  <TermRendering Id="אֲחַשְׁוֵרוֹשׁ" Guess="false">
    <Renderings>Xerxes</Renderings>
    <Glossary />
    <Changes />
    <Notes />
    <Denials />
  </TermRendering>
</TermRenderingsList>
""",
        }
    )
    terms: List[Tuple[str, List[str]]] = env.get_glosses()
    assert len(terms) == 1

    glosses = terms[0][1]
    assert str.join(" ", glosses) == "Xerxes"


def test_get_key_terms_from_terms_localizations_no_term_renderings() -> None:
    env = _TestEnvironment(
        _DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
        use_term_glosses=True,
    )
    terms: List[Tuple[str, List[str]]] = env.get_glosses()
    assert len(terms) == 5726

    glosses = terms[0][1]
    assert str.join(" ", glosses) == "Abagtha"


def test_get_key_terms_from_terms_localizations_no_term_renderings_do_not_use_term_glosses() -> None:
    env = _TestEnvironment(
        _DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
        use_term_glosses=False,
    )
    terms: List[Tuple[str, List[str]]] = env.get_glosses()
    assert len(terms) == 0


def test_get_key_terms_from_terms_localizations() -> None:
    env = _TestEnvironment(
        _DefaultParatextProjectSettings(
            biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml", language_code="fr"
        ),
        use_term_glosses=True,
    )
    terms: List[Tuple[str, List[str]]] = env.get_glosses()
    assert len(terms) == 5715

    glosses = terms[0][1]
    assert str.join(" ", glosses) == "Aaron"


def test_get_key_terms_from_terms_localizations_term_renderings_exists_prefer_localization() -> None:
    env = _TestEnvironment(
        _DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
        files={
            "TermRenderings.xml": r"""
<TermRenderingsList>
  <TermRendering Id="אֲחַשְׁוֵרוֹשׁ" Guess="false">
    <Renderings>Xerxes</Renderings>
    <Glossary />
    <Changes />
    <Notes />
    <Denials />
  </TermRendering>
</TermRenderingsList>
"""
        },
        use_term_glosses=True,
    )
    terms: List[Tuple[str, List[str]]] = env.get_glosses()
    assert len(terms) == 5726

    terms_index1_glosses = terms[1][1]
    terms_index2_glosses = terms[2][1]
    assert str.join(" ", terms_index1_glosses) == "Abagtha"
    assert str.join(" ", terms_index2_glosses) == "Abi"


def test_strip_parens() -> None:
    assert _strip_parens("") == ""
    assert _strip_parens("(inside)") == ""
    assert _strip_parens("Outside (inside)") == "Outside "
    assert _strip_parens("(Inside (inside)) Outside (Inside) (") == " Outside  ("
    assert _strip_parens("[inside] (outside)", "[", "]") == " (outside)"


def test_get_glosses() -> None:
    assert _get_glosses("") == []
    assert _get_glosses("*Abba* /") == ["Abba"]
    assert _get_glosses("Abba|| ") == ["Abba"]
    assert _get_glosses("Abba||Abbah?") == ["Abba", "Abbah"]
    assert _get_glosses("Abba (note)") == ["Abba"]
    assert _get_glosses("Ahasuerus, Xerxes; Assuerus") == ["Ahasuerus", "Xerxes", "Assuerus"]


class _TestEnvironment:
    def __init__(
        self,
        settings: Optional[ParatextProjectSettings] = None,
        files: Optional[Dict[str, str]] = None,
        use_term_glosses: bool = True,
    ) -> None:
        self._use_term_glosses: bool = use_term_glosses
        self._parser: ParatextProjectTermsParserBase = MemoryParatextProjectTermsParser(
            settings or _DefaultParatextProjectSettings(), files or {}
        )

    @property
    def parser(self) -> ParatextProjectTermsParserBase:
        return self._parser

    def get_glosses(self) -> List[Tuple[str, List[str]]]:
        return self.parser.parse(["PN"], self._use_term_glosses)


class _DefaultParatextProjectSettings(ParatextProjectSettings):
    def __init__(
        self,
        name: str = "Test",
        full_name: str = "TestProject",
        encoding: Optional[str] = None,
        versification: Optional[Versification] = None,
        stylesheet: Optional[UsfmStylesheet] = None,
        file_name_prefix: str = "",
        file_name_form: str = "41MAT",
        file_name_suffix: str = "Test.SFM",
        biblical_terms_list_type: str = "Project",
        biblical_terms_project_name: str = "Test",
        biblical_terms_file_name: str = "ProjectBiblicalTerms.xml",
        language_code: str = "en",
    ):

        super().__init__(
            name,
            full_name,
            encoding if encoding is not None else "utf-8",
            versification if versification is not None else ORIGINAL_VERSIFICATION,
            stylesheet if stylesheet is not None else UsfmStylesheet("usfm.sty"),
            file_name_prefix,
            file_name_form,
            file_name_suffix,
            biblical_terms_list_type,
            biblical_terms_project_name,
            biblical_terms_file_name,
            language_code,
        )
