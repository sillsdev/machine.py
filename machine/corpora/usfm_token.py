from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import regex as re

from .usfm_marker import UsfmMarker


class UsfmTokenType(Enum):
    BOOK = auto()
    CHAPTER = auto()
    VERSE = auto()
    TEXT = auto()
    PARAGRAPH = auto()
    CHARACTER = auto()
    NOTE = auto()
    END = auto()
    MILESTONE = auto()
    MILESTONE_END = auto()
    ATTRIBUTE = auto()
    UNKNOWN = auto()


_ATTRIBUTE_STR = r"([-\w]+)\s*\=\s*\"(.+?)\"\s*"
_ATTRIBUTE_REGEX = re.compile(_ATTRIBUTE_STR)
_ATTRIBUTES_REGEX = re.compile(r"(?<full>(" + _ATTRIBUTE_STR + r")+)|(?<default>[^\\=|]*)")


@dataclass
class UsfmToken:
    type: UsfmTokenType
    marker: Optional[UsfmMarker]
    text: Optional[str]
    data: Optional[str] = None
    is_nested: bool = False
    col_span: int = 0

    def __after_init__(self) -> None:
        self._default_attribute_name: Optional[str] = None
        self._attributes: Optional[List[_NamedAttribute]] = None

    def get_attribute(self, name: str) -> str:
        if self._attributes is None or len(self._attributes) == 0:
            return ""

        attribute = next((a for a in self._attributes if a.name == name), None)
        if attribute is None:
            return ""
        return attribute.value

    def set_attributes(
        self, attributes_value: str, default_attribute_name: Optional[str], text: str, preserve_whitespace: bool = False
    ) -> Optional[str]:
        if attributes_value is None or attributes_value == "" or self.marker is None:
            return None

        # for figures, convert 2.0 format to 3.0 format. Will need to write this as the 2.0 format
        # if the project is not upgrated.
        if self.marker.tag == "fig" and sum(1 for c in attributes_value if c == "|") == 5:
            attribute_list: List[_NamedAttribute] = []
            parts = attributes_value.split("|")
            _append_attribute(attribute_list, "alt", text)
            _append_attribute(attribute_list, "src", parts[0])
            _append_attribute(attribute_list, "size", parts[1])
            _append_attribute(attribute_list, "loc", parts[2])
            _append_attribute(attribute_list, "copy", parts[3])
            whitespace = ""
            if preserve_whitespace:
                whitespace = text[: len(text) - len(text.lstrip())]
            text = whitespace + parts[4]
            _append_attribute(attribute_list, "ref", parts[5])
            self._attributes = attribute_list
            return text

        match = _ATTRIBUTES_REGEX.match(attributes_value)
        if match is None or len(match.group()) != len(attributes_value):
            return None

        default_value = match.group("default")
        if default_value is not None:
            if default_attribute_name is not None:
                self._attributes = [_NamedAttribute(default_attribute_name, default_value)]
                self._default_attribute_name = default_attribute_name
                self._is_default_attribute = True
                return text
            return None

        full = match.group("full")
        if full is None:
            return None

        self._default_attribute_name = default_attribute_name
        self._attributes = []
        for i, attr_match in enumerate(_ATTRIBUTE_REGEX.findall(full)):
            self._attributes.append(_NamedAttribute(attr_match[0], attr_match[1], i))
        return text

    def copy_attributes(self, source_token: "UsfmToken") -> None:
        self._attributes = source_token._attributes
        self._default_attribute_name = source_token._default_attribute_name

    def __repr__(self) -> str:
        string = ""
        if self.type == UsfmTokenType.ATTRIBUTE:
            string += f"|{self.data}"
        else:
            if self.marker is not None:
                string += f"\\+{self.marker.tag}" if self.is_nested else str(self.marker)
                if self.col_span >= 2:
                    col = int(self.marker.tag[-1])
                    string += f"-{col + self.col_span - 1}"
            if self.text is not None and self.text != "":
                if len(string) > 0:
                    string += " "
                string += self.text
            if self.data is not None and self.data != "":
                if len(string) > 0:
                    string += " "
                string += self.data
        return string


@dataclass
class _NamedAttribute:
    name: str
    value: str
    offset: int = 0

    def __repr__(self) -> str:
        return f'{self.name}="{self.value}"'


def _append_attribute(attributes: List[_NamedAttribute], name: str, value: str) -> None:
    value = value.strip()
    if value is not None and value != "":
        attributes.append(_NamedAttribute(name, value))
