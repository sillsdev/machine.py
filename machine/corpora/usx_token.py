from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree


@dataclass(frozen=True)
class UsxToken:
    para_element: ElementTree.Element
    text: str
    element: Optional[ElementTree.Element]

    def __repr__(self) -> str:
        return self.text
