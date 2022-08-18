from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence

import regex as re


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
class UsfmAttribute:
    name: str
    value: str
    offset: int = 0

    def __repr__(self) -> str:
        return f'{self.name}="{self.value}"'


@dataclass
class UsfmToken:
    type: UsfmTokenType
    marker: Optional[str]
    text: Optional[str]
    end_marker: Optional[str]
    data: Optional[str] = None

    @property
    def nestless_marker(self) -> Optional[str]:
        return self.marker[1:] if self.marker is not None and self.marker[0] == "+" else self.marker

    def __post_init__(self) -> None:
        self.attributes: Optional[Sequence[UsfmAttribute]] = None
        self._default_attribute_name: Optional[str] = None

    def get_attribute(self, name: str) -> str:
        if self.attributes is None or len(self.attributes) == 0:
            return ""

        attribute = next((a for a in self.attributes if a.name == name), None)
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
        if self.nestless_marker == "fig" and sum(1 for c in attributes_value if c == "|") == 6:
            attribute_list: List[UsfmAttribute] = []
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
            self.attributes = attribute_list
            return text

        match = _ATTRIBUTES_REGEX.match(attributes_value)
        if match is None or len(match.group()) != len(attributes_value):
            return None

        default_value = match.group("default")
        if default_value is not None:
            if default_attribute_name is not None:
                self.attributes = [UsfmAttribute(default_attribute_name, default_value)]
                self._default_attribute_name = default_attribute_name
                self._is_default_attribute = True
                return text
            return None

        full = match.group("full")
        if full is None:
            return None

        self._default_attribute_name = default_attribute_name
        self.attributes = []
        for i, attr_match in enumerate(_ATTRIBUTE_REGEX.findall(full)):
            self.attributes.append(UsfmAttribute(attr_match[0], attr_match[1], i))
        return text

    def copy_attributes(self, source_token: UsfmToken) -> None:
        self.attributes = source_token.attributes
        self._default_attribute_name = source_token._default_attribute_name

    def get_length(self, include_newlines: bool = False, add_spaces: bool = True) -> int:
        # WARNING: This logic in this method needs to match the logic in to_usfm()

        total_length = len(self.text) if self.text is not None else 0
        if self.type == UsfmTokenType.ATTRIBUTE:
            total_length += len(self.to_attribute_string())
        elif self.marker is not None:
            if include_newlines and self.type in {UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE}:
                total_length += 2
            total_length += len(self.marker) + 1  # marker and backslash
            if add_spaces and (len(self.marker) == 0 or self.marker[-1] != "*"):
                total_length += 1  # space

            if self.data is not None and len(self.data) > 0:
                if len(self.marker) > 0:
                    total_length += 1
                total_length += len(self.data)
                if add_spaces:
                    total_length += 1

            if self.type in {UsfmTokenType.MILESTONE, UsfmTokenType.MILESTONE_END}:
                attributes = self.to_attribute_string()
                if len(attributes) > 0:
                    total_length += len(attributes)
                else:
                    # remove space that was put after marker - not needed when there are no attributes.
                    total_length -= 1

                total_length += 2  # End of the milestone
        return total_length

    def to_usfm(self, include_newlines: bool = False, add_spaces: bool = True) -> str:
        # WARNING: This logic in this method needs to match the logic in get_length()

        to_return = self.text if self.text is not None else ""
        if self.type == UsfmTokenType.ATTRIBUTE:
            to_return += self.to_attribute_string()
        elif self.marker is not None:
            if include_newlines and self.type in {UsfmTokenType.PARAGRAPH, UsfmTokenType.CHAPTER, UsfmTokenType.VERSE}:
                to_return += "\r\n"
            to_return += "\\"
            if len(self.marker) > 0:
                to_return += self.marker
            if add_spaces and (len(self.marker) == 0 or self.marker[-1] != "*"):
                to_return += " "

            if self.data is not None and len(self.data) > 0:
                if len(self.marker) > 0:
                    to_return += " "
                to_return += self.data
                if add_spaces:
                    to_return += " "

            if self.type in {UsfmTokenType.MILESTONE, UsfmTokenType.MILESTONE_END}:
                attributes = self.to_attribute_string()
                if len(attributes) > 0:
                    to_return += attributes
                else:
                    # remove space that was put after marker - not needed when there are no attributes.
                    to_return = to_return[:-1]
                to_return += "\\*"
        return to_return

    def to_attribute_string(self) -> str:
        if self.attributes is None or len(self.attributes) == 0:
            return ""

        if self.data is not None:
            return "|" + self.data

        if len(self.attributes) == 1 and self.attributes[0].name == self._default_attribute_name:
            return "|" + self.attributes[0].value

        return "|" + (" ".join(str(a) for a in self.attributes))

    def __repr__(self) -> str:
        return self.to_usfm(add_spaces=False)


def _append_attribute(attributes: List[UsfmAttribute], name: str, value: str) -> None:
    value = value.strip()
    if value is not None and value != "":
        attributes.append(UsfmAttribute(name, value))
