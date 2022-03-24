from typing import FrozenSet, Generic, Iterable, Optional, TypeVar, overload

T = TypeVar("T")


class Cluster(Generic[T]):
    @overload
    def __init__(self, *data_objects: T, noise: bool = False, description: Optional[str] = None) -> None:
        ...

    @overload
    def __init__(self, data_objects: Iterable[T], noise: bool = False, description: Optional[str] = None) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        self._data_objects: FrozenSet[T]
        self._noise: bool
        self._description: Optional[str]
        if len(args) == 0:
            self._data_objects = kwargs.get("data_objects", frozenset())
            self._noise = kwargs.get("noise", False)
            self.description = kwargs.get("description", None)
        elif not isinstance(args[0], str) and isinstance(args[0], Iterable):
            self._data_objects = frozenset(args[0])
            self._noise = args[1] if len(args) >= 2 else kwargs.get("noise", False)
            self.description = args[2] if len(args) >= 3 else kwargs.get("description", None)
        else:
            self._data_objects = frozenset(args)
            self._noise: bool = kwargs.get("noise", False)
            self.description = kwargs.get("description", None)

    @property
    def data_objects(self) -> FrozenSet[T]:
        return self._data_objects

    @property
    def noise(self) -> bool:
        return self._noise

    def __repr__(self) -> str:
        if self.description is not None:
            return self.description
        return "EMPTY" if len(self._data_objects) == 0 else str(self._data_objects)
