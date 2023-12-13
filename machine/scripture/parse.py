import re
from typing import Dict, List, Set, Union

from .canon import book_id_to_number
from .constants import ORIGINAL_VERSIFICATION
from .verse_ref import Versification

COMMA_SEPARATED_BOOKS = re.compile(r"([A-Z\d]{3}|OT|NT)(, ?([A-Z\d]{3}|OT|NT))*")
BOOK_RANGE = re.compile(r"[A-Z\d]{3}-[A-Z\d]{3}")
CHAPTER_SELECTION = re.compile(r"[A-Z\d]{3} ?(\d+|\d+-\d+)(, ?(\d+|\d+-\d+))*")


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
def get_chapters(
    chapter_selections: Union[str, List[str]], versification: Versification = ORIGINAL_VERSIFICATION
) -> Dict[int, List[int]]:
    chapters = {}

    if isinstance(chapter_selections, str):
        chapter_selections = chapter_selections.strip()

        if ";" in chapter_selections:
            chapter_selections = chapter_selections.split(";")
        elif re.fullmatch(COMMA_SEPARATED_BOOKS, chapter_selections) is not None:
            chapter_selections = chapter_selections.split(",")
        elif chapter_selections.startswith("-"):
            raise ValueError(f"Cannot subtract before adding sections: {chapter_selections}")
        elif re.search(BOOK_RANGE, chapter_selections):
            if len(chapter_selections) == 7:
                chapter_selections = [chapter_selections]
            else:
                raise ValueError(
                    "Invalid syntax. If one of your selections is a range of books, \
                    selections must be seprated with semicolons."
                )
        elif re.fullmatch(CHAPTER_SELECTION, chapter_selections) is None:
            raise ValueError(
                "Invalid syntax. If one of your selections includes specific chapters or subtraction, \
                selections must be separated with semicolons."
            )
        else:
            chapter_selections = [chapter_selections]

    for section in chapter_selections:
        section = section.strip()

        if section.startswith("-"):  # Subtraction
            section = section[1:]
            if any(
                s.isdigit() and (i == len(section) - 1 or not section[i + 1].isalpha()) for i, s in enumerate(section)
            ):  # Specific chapters from one book
                book = book_id_to_number(section[:3])
                if book == 0:
                    raise ValueError(f"{section[:3]} is an invalid book ID.")
                if book not in chapters:
                    raise ValueError(f"{section[:3]} cannot be removed as it is not in the existing book selection.")

                if chapters[book] == []:
                    chapters[book] = [i + 1 for i in range(versification.get_last_chapter(book))]

                last_chapter = versification.get_last_chapter(book)
                chapter_nums = section[3:].split(",")
                for chapter_num in chapter_nums:
                    chapter_num = chapter_num.strip()
                    if "-" in chapter_num:
                        start, end = chapter_num.split("-")
                        if int(start) > last_chapter or int(end) > last_chapter:
                            raise ValueError(f"{chapter_num} is an invalid chapter range.")

                        for i in range(int(start), int(end) + 1):
                            if i not in chapters[book]:
                                raise ValueError(
                                    f"{i} cannot be removed as it is not in the existing chapter selection."
                                )
                            chapters[book].remove(i)
                    else:
                        if int(chapter_num) > last_chapter:
                            raise ValueError(f"{int(chapter_num)} is an invalid chapter number.")
                        if int(chapter_num) not in chapters[book]:
                            raise ValueError(
                                f"{chapter_num} cannot be removed as it is not in the existing chapter selection."
                            )
                        chapters[book].remove(int(chapter_num))

                if len(chapters[book]) == 0:
                    del chapters[book]
            elif "-" in section:  # Spans of books
                ends = section.split("-")
                if len(ends) != 2 or book_id_to_number(ends[0]) == 0 or book_id_to_number(ends[1]) == 0:
                    raise ValueError(f"{section} is an invalid book range.")

                for i in range(book_id_to_number(ends[0]), book_id_to_number(ends[1]) + 1):
                    if i not in chapters:
                        raise ValueError(
                            f"{section[:3]} cannot be removed as it is not in the existing book selection."
                        )

                    del chapters[i]
            else:  # Single whole book
                book = book_id_to_number(section)
                if book == 0:
                    raise ValueError(f"{section} is an invalid book ID.")
                if book not in chapters:
                    raise ValueError(f"{section[:3]} cannot be removed as it is not in the existing book selection.")

                del chapters[book]
        elif any(
            s.isdigit() and (i == len(section) - 1 or not section[i + 1].isalpha()) for i, s in enumerate(section)
        ):  # Specific chapters from one book
            book = book_id_to_number(section[:3])
            if book == 0:
                raise ValueError(f"{section[:3]} is an invalid book ID.")

            if book in chapters:
                if chapters[book] == []:
                    continue
                book_chapters = set(chapters[book])
            else:
                book_chapters = set()

            last_chapter = versification.get_last_chapter(book)
            chapter_nums = section[3:].split(",")
            for chapter_num in chapter_nums:
                chapter_num = chapter_num.strip()
                if "-" in chapter_num:
                    start, end = chapter_num.split("-")
                    if int(start) > last_chapter or int(end) > last_chapter:
                        raise ValueError(f"{chapter_num} is an invalid chapter range.")

                    for i in range(int(start), int(end) + 1):
                        book_chapters.add(i)
                else:
                    if int(chapter_num) > last_chapter:
                        raise ValueError(f"{chapter_num} is an invalid chapter number.")

                    book_chapters.add(int(chapter_num))

            if len(book_chapters) == last_chapter:
                chapters[book] = []
            else:
                chapters[book] = sorted(list(book_chapters))
        elif "-" in section:  # Spans of books
            ends = section.split("-")
            if len(ends) != 2 or book_id_to_number(ends[0]) == 0 or book_id_to_number(ends[1]) == 0:
                raise ValueError(f"{section} is an invalid book range.")

            for i in range(book_id_to_number(ends[0]), book_id_to_number(ends[1]) + 1):
                chapters[i] = []
        elif section == "OT":
            for i in range(1, 40):
                chapters[i] = []
        elif section == "NT":
            for i in range(40, 67):
                chapters[i] = []
        else:  # Single whole book
            book = book_id_to_number(section)
            if book == 0:
                raise ValueError(f"{section} is an invalid book ID.")
            chapters[book] = []

    return chapters
