from typing import Any, ContextManager, Generator, TypeVar

T_co = TypeVar("T_co")
T_contra = TypeVar("T_contra")
V_co = TypeVar("V_co")


class ContextManagedGenerator(ContextManager[Generator[T_co, T_contra, V_co]], Generator[T_co, T_contra, V_co]):
    def __init__(self, generator: Generator[T_co, T_contra, V_co]) -> None:
        self._generator = generator

    def __next__(self) -> T_co:
        return self._generator.__next__()

    def __iter__(self) -> Generator[T_co, T_contra, V_co]:
        return self._generator.__iter__()

    def send(self, value: T_contra) -> T_co:
        return self._generator.send(value)

    def throw(self, type: Any, value: Any = None, traceback: Any = None) -> T_co:
        return self._generator.throw(type, value, traceback)

    def close(self) -> None:
        self._generator.close()

    def __enter__(self) -> Generator[T_co, T_contra, V_co]:
        return super().__enter__()

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()
