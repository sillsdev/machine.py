import xml.etree.ElementTree as etree
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UsxToken:
    para_element: etree.Element
    text: str
    element: Optional[etree.Element]

    def __repr__(self) -> str:
        return self.text
