import re
from typing import Dict, List, Set, Union

from .canon import book_id_to_number, book_number_to_id
from .constants import ORIGINAL_VERSIFICATION
from .verse_ref import Versification

COMMA_SEPARATED_BOOKS = re.compile(r"([A-Z\d]{3}|OT|NT)(, ?([A-Z\d]{3}|OT|NT))*")
BOOK_RANGE = re.compile(r"-?[A-Z\d]{3}-[A-Z\d]{3}")
CHAPTER_SELECTION = re.compile(r"-?[A-Z\d]{3} ?(\d+|\d+-\d+)(, ?(\d+|\d+-\d+))*")


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


def parse_selection(selection: str, versification: Versification) -> Dict[int, List[int]]:
    selection = selection.strip()
    chapters = {}

    if selection[-1].isdigit() and len(selection) > 3:  # Specific chapters from one book
        book = book_id_to_number(selection[:3])
        if book == 0:
            raise ValueError(f"{selection[:3]} is an invalid book ID.")

        book_chapters = set()
        last_chapter = versification.get_last_chapter(book)
        chapter_nums = selection[3:].split(",")
        for chapter_num in chapter_nums:
            chapter_num = chapter_num.strip()
            if "-" in chapter_num:
                start, end = chapter_num.split("-")
                if int(start) == 0 or int(end) > last_chapter or int(end) <= int(start):
                    raise ValueError(f"{chapter_num} is an invalid chapter range.")

                for i in range(int(start), int(end) + 1):
                    book_chapters.add(i)
            else:
                if int(chapter_num) == 0 or int(chapter_num) > last_chapter:
                    raise ValueError(f"{chapter_num} is an invalid chapter number.")

                book_chapters.add(int(chapter_num))

        if len(book_chapters) == last_chapter:
            chapters[book] = []
        else:
            chapters[book] = sorted(list(book_chapters))
    elif "-" in selection:  # Span of books
        ends = selection.split("-")
        if len(ends) != 2 or book_id_to_number(ends[0]) == 0 or book_id_to_number(ends[1]) == 0:
            raise ValueError(f"{selection} is an invalid book range.")

        if book_id_to_number(ends[0]) >= book_id_to_number(ends[1]):
            raise ValueError(f"{selection} is an invalid book range. {ends[1]} precedes {ends[0]}.")

        for i in range(book_id_to_number(ends[0]), book_id_to_number(ends[1]) + 1):
            chapters[i] = []
    elif selection == "OT":
        for i in range(1, 40):
            chapters[i] = []
    elif selection == "NT":
        for i in range(40, 67):
            chapters[i] = []
    else:  # Single whole book
        book = book_id_to_number(selection)
        if book == 0:
            raise ValueError(f"{selection} is an invalid book ID.")
        chapters[book] = []

    return chapters


# Output format: { book_num: [chapters] }
# An empty list, i.e. book_num: [] signifies the inclusion of all chapters
def get_chapters(
    selections: Union[str, List[str]], versification: Versification = ORIGINAL_VERSIFICATION
) -> Dict[int, List[int]]:
    chapters = {}
    if isinstance(selections, str):
        selections = selections.strip()

        if len(selections) == 0:
            return chapters

        delimiter = ";"
        if ";" in selections:
            delimiter = ";"
        elif re.fullmatch(COMMA_SEPARATED_BOOKS, selections) is not None:
            delimiter = ","
        elif re.fullmatch(BOOK_RANGE, selections) is None and re.fullmatch(CHAPTER_SELECTION, selections) is None:
            raise ValueError(
                "Invalid syntax. If you are providing multiple selections, e.g. a range of books followed by a"
                "selection of chapters from a book, separate each selection with a semicolon."
            )

        selections = selections.split(delimiter)

    for selection in selections:
        selection = selection.strip()

        if selection.startswith("-"):  # Subtraction
            selection_chapters = parse_selection(selection[1:], versification)
            for book in selection_chapters:
                if book not in chapters:
                    raise ValueError(
                        f"{book_number_to_id(book)} cannot be removed as it is not in the existing book selection."
                    )

                if selection_chapters[book] == []:
                    selection_chapters[book] = [i + 1 for i in range(versification.get_last_chapter(book))]

                if chapters[book] == []:
                    chapters[book] = [i + 1 for i in range(versification.get_last_chapter(book))]

                for chapter in selection_chapters[book]:
                    try:
                        chapters[book].remove(chapter)
                    except ValueError:
                        raise ValueError(
                            f"{book_number_to_id(book)} {chapter} cannot be removed as it is not in the existing"
                            "chapter selection."
                        )

                if len(chapters[book]) == 0:
                    del chapters[book]
        else:
            selection_chapters = parse_selection(selection, versification)
            for book in selection_chapters:
                if book in chapters:
                    if chapters[book] == [] or selection_chapters[book] == []:
                        chapters[book] = []
                        continue

                    chapters[book] = list(set(chapters[book] + selection_chapters[book]))
                    if len(chapters[book]) == versification.get_last_chapter(book):
                        chapters[book] = []
                else:
                    chapters[book] = selection_chapters[book]

    return chapters
