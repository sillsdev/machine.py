from collections.abc import Mapping, Set
from typing import Any, Generic, Iterable, Iterator, Optional, Tuple, TypeVar, overload

from .graph import Graph

T = TypeVar("T")

class NodeView(Mapping[Any, Any], Set[T]):
    def __init__(self, graph: Graph[T]) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Iterator[T]: ...
    def __getitem__(self, n: Any): ...
    def __contains__(self, n: Any): ...
    def __call__(self, data: bool = ..., default: Optional[Any] = ...) -> NodeView[T]: ...
    def data(self, data: bool = ..., default: Optional[Any] = ...) -> NodeDataView[T]: ...

class NodeDataView(Set[T], Iterable[T]):
    def __init__(self, nodedict: Any, data: bool = ..., default: Optional[Any] = ...) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Iterator[T]: ...
    def __contains__(self, n: T): ...
    def __getitem__(self, n: Any): ...

class DiDegreeView(Mapping[T, int], Iterable[int]):
    def __init__(self, G: Any, nbunch: Optional[Any] = ..., weight: Optional[Any] = ...) -> None: ...
    def __call__(self, nbunch: Optional[Any] = ..., weight: Optional[Any] = ...): ...
    def __iter__(self) -> Iterator[Tuple[T, int]]: ...
    def __getitem__(self, n: T): ...
    def __len__(self): ...

class DegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...

class OutDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: T): ...

class InDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...

class MultiDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...

class DiMultiDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...

class InMultiDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...
    def __iter__(self) -> None: ...

class OutMultiDegreeView(DiDegreeView[T]):
    def __getitem__(self, n: Any): ...
    def __iter__(self) -> None: ...

class OutEdgeDataView(Generic[T]):
    def __init__(
        self, viewer: Any, nbunch: Optional[Any] = ..., data: bool = ..., default: Optional[Any] = ...
    ) -> None: ...
    def __len__(self): ...
    def __iter__(self): ...
    def __contains__(self, e: Any): ...

class EdgeDataView(OutEdgeDataView[T]):
    def __len__(self): ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...

class InEdgeDataView(OutEdgeDataView[T]):
    def __iter__(self): ...
    def __contains__(self, e: Any): ...

class OutMultiEdgeDataView(OutEdgeDataView[T]):
    keys: Any = ...
    def __init__(
        self, viewer: Any, nbunch: Optional[Any] = ..., data: bool = ..., keys: bool = ..., default: Optional[Any] = ...
    ) -> None: ...
    def __len__(self): ...
    def __iter__(self): ...
    def __contains__(self, e: Any): ...

class MultiEdgeDataView(OutMultiEdgeDataView[T]):
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...

class InMultiEdgeDataView(OutMultiEdgeDataView[T]):
    def __iter__(self): ...
    def __contains__(self, e: Any): ...

class OutEdgeView(Set[T], Mapping[Any, T]):
    dataview: Any = ...
    def __init__(self, G: Any) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...
    def __getitem__(self, e: Any): ...
    @overload
    def __call__(self, nbunch: T = ..., data: bool = ..., default: Optional[Any] = ...) -> OutEdgeDataView[T]: ...
    @overload
    def __call__(
        self, nbunch: Optional[Any] = ..., data: bool = ..., default: Optional[Any] = ...
    ) -> OutEdgeDataView: ...
    def data(self, data: bool = ..., default: Optional[Any] = ..., nbunch: Optional[Any] = ...): ...

class EdgeView(OutEdgeView[T]):
    dataview: Any = ...
    def __len__(self): ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...

class InEdgeView(OutEdgeView[T]):
    dataview: Any = ...
    def __init__(self, G: Any) -> None: ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...
    def __getitem__(self, e: Any): ...
    def __call__(self, nbunch: T = ..., data: bool = ..., default: Optional[Any] = ...) -> InEdgeDataView: ...

class OutMultiEdgeView(OutEdgeView[T]):
    dataview: Any = ...
    def __len__(self): ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...
    def __getitem__(self, e: Any): ...
    def __call__(
        self, nbunch: Optional[Any] = ..., data: bool = ..., keys: bool = ..., default: Optional[Any] = ...
    ): ...
    def data(self, data: bool = ..., keys: bool = ..., default: Optional[Any] = ..., nbunch: Optional[Any] = ...): ...

class MultiEdgeView(OutMultiEdgeView[T]):
    dataview: Any = ...
    def __len__(self): ...
    def __iter__(self) -> None: ...

class InMultiEdgeView(OutMultiEdgeView[T]):
    dataview: Any = ...
    def __init__(self, G: Any) -> None: ...
    def __iter__(self) -> None: ...
    def __contains__(self, e: Any): ...
    def __getitem__(self, e: Any): ...