from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, TypeVar


T = TypeVar("T")


def dataclass_ex(_cls: Optional[Type[T]] = None, *args: Any, **kwargs: Any) -> Callable[[Type[T]], Type[T]]:
    def wrap(cls: Type[T]) -> Type[T]:
        # Save the current __init__ and remove it so dataclass will
        # create the default __init__.
        user_init = getattr(cls, "__init__")
        delattr(cls, "__init__")

        # let dataclass process our class.
        result = dataclass(cls, *args, **kwargs)

        # Restore the user's __init__ save the default init to __default_init__.
        setattr(result, "__default_init__", result.__init__)
        setattr(result, "__init__", user_init)

        # Just in case that dataclass will return a new instance,
        # (currently, does not happen), restore cls's __init__.
        if result is not cls:
            setattr(cls, "__init__", user_init)

        return result

    # Support both dataclass_ex() and dataclass_ex
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)


def init_dataclass_ex(self: Any, *args: Any, **kwargs: Any) -> None:
    self.__default_init__(*args, **kwargs)
