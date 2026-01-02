from typing import Dict, List, Optional, Sequence, Tuple

from testutils.memory_paratext_project_file_handler import DefaultParatextProjectSettings
from testutils.memory_paratext_project_terms_parser import MemoryParatextProjectTermsParser

from machine.corpora import KeyTerm, ParatextProjectSettings, ParatextProjectTermsParserBase
from machine.corpora.paratext_project_terms_parser_base import _get_glosses, _get_renderings_with_pattern, _strip_parens


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
    terms: List[KeyTerm] = env.get_glosses()
    assert len(terms) == 1

    glosses = terms[0].renderings
    assert str.join(" ", glosses) == "Xerxes"


def test_get_key_terms_from_terms_localizations_no_term_renderings() -> None:
    env = _TestEnvironment(
        DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
        use_term_glosses=True,
    )
    terms: List[KeyTerm] = env.get_glosses()
    assert len(terms) == 5726

    glosses = terms[0].renderings
    assert str.join(" ", glosses) == "Aaron"


def test_get_key_terms_from_terms_localizations_no_term_renderings_do_not_use_term_glosses() -> None:
    env = _TestEnvironment(
        DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
        use_term_glosses=False,
    )
    terms: List[KeyTerm] = env.get_glosses()
    assert len(terms) == 0


def test_get_key_terms_from_terms_localizations() -> None:
    env = _TestEnvironment(
        DefaultParatextProjectSettings(
            biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml", language_code="fr"
        ),
        use_term_glosses=True,
    )
    terms: List[KeyTerm] = env.get_glosses()
    assert len(terms) == 5715

    glosses = terms[0].renderings
    assert str.join(" ", glosses) == "Aaron"


def test_get_key_terms_from_terms_localizations_term_renderings_exists_prefer_localization() -> None:
    env = _TestEnvironment(
        DefaultParatextProjectSettings(biblical_terms_list_type="Major", biblical_terms_file_name="BiblicalTerms.xml"),
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
    terms: List[KeyTerm] = env.get_glosses()
    assert len(terms) == 5726

    terms_index1_glosses = terms[1].renderings
    terms_index2_glosses = terms[2].renderings
    assert str.join(" ", terms_index1_glosses) == "Obadiah"
    assert str.join(" ", terms_index2_glosses) == "Abagtha"


def test_strip_parens() -> None:
    assert _strip_parens("") == ""
    assert _strip_parens("(inside)") == ""
    assert _strip_parens("Outside (inside)") == "Outside "
    assert _strip_parens("(Inside (inside)) Outside (Inside) (") == " Outside  ("
    assert _strip_parens("[inside] (outside)", "[", "]") == " (outside)"


def test_get_glosses() -> None:
    assert _get_glosses("") == []
    assert _get_glosses("Abba (note)") == ["Abba"]
    assert set(_get_glosses("Ahasuerus, Xerxes; Assuerus")) == set(["Assuerus", "Xerxes", "Ahasuerus"])


def test_get_renderings() -> None:
    assert _get_renderings_with_pattern("") == []
    assert _get_renderings_with_pattern("*Abba*") == ["*Abba*"]
    assert _get_renderings_with_pattern("Abba|| ") == ["Abba"]
    assert _get_renderings_with_pattern("Abba||Abbah") == ["Abba", "Abbah"]
    assert _get_renderings_with_pattern("Abba (note)") == ["Abba"]


class _TestEnvironment:
    def __init__(
        self,
        settings: Optional[ParatextProjectSettings] = None,
        files: Optional[Dict[str, str]] = None,
        use_term_glosses: bool = True,
    ) -> None:
        self._use_term_glosses: bool = use_term_glosses
        self._parser: ParatextProjectTermsParserBase = MemoryParatextProjectTermsParser(
            files or {}, settings or DefaultParatextProjectSettings()
        )

    @property
    def parser(self) -> ParatextProjectTermsParserBase:
        return self._parser

    def get_glosses(self) -> List[KeyTerm]:
        return list(self.parser.parse(["PN"], self._use_term_glosses))
