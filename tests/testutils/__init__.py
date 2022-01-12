from abc import ABC
from pathlib import Path
from typing import Type, TypeVar, cast

T = TypeVar("T", bound=ABC)

TEST_DATA_PATH = Path(__file__).parent / "data"


def make_concrete(abc_class: Type[T]) -> Type[T]:
    if "__abstractmethods__" not in abc_class.__dict__:
        return abc_class
    new_dict = abc_class.__dict__.copy()
    for abstractmethod in abc_class.__abstractmethods__:
        # replace each abc method or property with an identity function:
        new_dict[abstractmethod] = lambda x, *args, **kw: (x, args, kw)
    # creates a new class, with the overriden ABCs:
    return cast(Type[T], type("dummy_concrete_%s" % abc_class.__name__, (abc_class,), new_dict))  # type: ignore
