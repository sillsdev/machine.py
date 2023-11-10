import re
from typing import List, Set, Union

from .canon import book_id_to_number
from .verse_ref import Versification

USFM_FILE_PATTERN = re.compile(r"(?<=[A-Z]{3})\d+\.usfm")
BOOK_SPAN = re.compile(r"[A-Z]{3}-[A-Z]{3}")


def get_books(books: Union[str, List[str]]) -> Set[int]:
    if isinstance(books, str):
        books = books.split(",")
    book_set: Set[int] = set()
    for book_id in books:
        book_id = book_id.strip().strip("*").upper()
        if book_id == "NT":
            book_set.update(range(40, 67))
        elif book_id == "OT":
            book_set.update(range(1, 40))
        elif book_id.startswith("-"):
            # remove the book from the set
            book_id = book_id[1:]
            book_num = book_id_to_number(book_id)
            if book_num == 0:
                raise RuntimeError(f"{book_id} is an invalid book ID.")
            elif book_num not in book_set:
                raise RuntimeError(
                    f"{book_id}:{book_num} cannot be removed as it is not in the existing book set of {book_set}"
                )
            else:
                book_set.remove(book_num)
        else:
            book_num = book_id_to_number(book_id)
            if book_num == 0:
                raise RuntimeError(f"{book_id} is an invalid book ID.")
            book_set.add(book_num)
    return book_set


# Output format: { book_num: [chapters] }
# An empty list, i.e. book_num: [] signifies the inclusion of all chapters
def get_chapters(chapter_selections: str) -> dict:
    versification = Versification.create("Original")
    chapters = {}
    spans = []
    subtractions = []

    # Normalize books written as "MAT01.usfm" to "MAT"
    chapter_selections = re.sub(USFM_FILE_PATTERN, "", chapter_selections)

    if ";" not in chapter_selections and not any(
        s.isdigit() and (i == len(chapter_selections) - 1 or not chapter_selections[i + 1].isalpha())
        for i, s in enumerate(chapter_selections)
    ):  # Backwards compatibility with get_books syntax:
        sections = chapter_selections.split(",")
    else:
        sections = chapter_selections.split(";")

    for section in sections:
        if section == "":
            continue
        elif section.startswith("-"):
            subtractions.append(section[1:])
        elif any(
            s.isdigit() and (i == len(section) - 1 or not section[i + 1].isalpha()) for i, s in enumerate(section)
        ):  # Specific chapters from one book
            book = book_id_to_number(section[:3])

            if book == 0:
                raise RuntimeError(f"{section[:3]} is an invalid book ID.")

            chapter_nums = section[3:].split(",")
            chapters[book] = set()
            last_chapter = versification.get_last_chapter(book)
            for chapter_num in chapter_nums:
                if "-" in chapter_num:
                    start, end = chapter_num.split("-")
                    for i in range(int(start), min(int(end), last_chapter) + 1):
                        chapters[book].add(i)
                elif int(chapter_num) <= last_chapter:
                    chapters[book].add(int(chapter_num))

            # Delete entry if no chapter numbers were valid
            if len(chapters[book]) == 0:
                del chapters[book]
        elif "-" in section:  # Spans of books
            spans.append(section)
        elif section == "OT":
            for i in range(1, 40):
                if i not in chapters:
                    chapters[i] = set()
        elif section == "NT":
            for i in range(40, 67):
                if i not in chapters:
                    chapters[i] = set()
        else:  # Single whole book
            book = book_id_to_number(section)
            if book == 0:
                raise RuntimeError(f"{section} is an invalid book ID.")

            if book not in chapters:
                chapters[book] = set()

    for span in spans:
        ends = span.split("-")
        if len(ends) != 2 or book_id_to_number(ends[0]) == 0 or book_id_to_number(ends[1]) == 0:
            raise RuntimeError(f"{span} is an invalid book range.")

        for i in range(book_id_to_number(ends[0]), book_id_to_number(ends[1]) + 1):
            if i not in chapters:
                chapters[i] = set()

    for subtraction in subtractions:
        if re.match(BOOK_SPAN, subtraction) is not None:
            raise RuntimeError("Cannot subtract spans of books.")

        book = book_id_to_number(subtraction[:3])
        if book == 0:
            raise RuntimeError(f"{subtraction[:3]} is an invalid book ID.")
        if book not in chapters:
            raise RuntimeError(f"{subtraction[:3]} cannot be removed as it is not in the existing book selection.")

        # Subtract entire book
        if len(subtraction) == 3:
            del chapters[book]
            continue

        if len(chapters[book]) == 0:
            chapters[book] = {i + 1 for i in range(versification.get_last_chapter(book))}
        chapter_nums = subtraction[3:].split(",")
        for chapter_num in chapter_nums:
            if "-" in chapter_num:
                start, end = chapter_num.split("-")
                for i in range(int(start), int(end) + 1):
                    chapters[book].discard(i)
            else:
                chapters[book].discard(int(chapter_num))

        # Delete entry if no chapter numbers are left
        if len(chapters[book]) == 0:
            del chapters[book]
        # Make entry the empty set again if all chapters are still present
        elif len(chapters[book]) == versification.get_last_chapter(book):
            chapters[book] = set()

    for k, v in chapters.items():
        chapters[k] = sorted(list(v))

    return chapters
