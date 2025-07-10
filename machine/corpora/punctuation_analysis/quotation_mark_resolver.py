from abc import ABC, abstractmethod
from typing import Generator, List, Set

from .quotation_mark_metadata import QuotationMarkMetadata
from .quotation_mark_resolution_issue import QuotationMarkResolutionIssue
from .quotation_mark_string_match import QuotationMarkStringMatch


class QuotationMarkResolver(ABC):

    @abstractmethod
    def resolve_quotation_marks(
        self, quotation_mark_matches: List[QuotationMarkStringMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_issues(self) -> Set[QuotationMarkResolutionIssue]: ...
