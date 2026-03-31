import copy
from functools import partial
from typing import Any, Callable
from collections.abc import Sequence

from parsl import File, python_app
from parsl.dataflow.futures import AppFuture, Future

from psiflow.geometry import Geometry


CLS_TO_IGNORE = (Future, Callable, File, Geometry)


def traverse(obj: Any, callback: Callable) -> Any:
    """
    Recursively traverses object and applies callback to every element.
    Returns the (potentially modified) structure.
    """
    node = callback(obj)
    match obj:
        case _ if isinstance(obj, CLS_TO_IGNORE):
            pass  # do not inspect certain classes
        case dict():
            return {k: traverse(v, callback) for k, v in node.items()}
        case list() | tuple() | set():
            return type(obj)(traverse(v, callback) for v in node)
        case object(__dict__=data):
            # filters for classes with attributes - will not work with slots
            for k, v in data.items():
                setattr(node, k, traverse(v, callback))
    return node


def extract_futures(obj: Any) -> list[AppFuture]:
    """Find all futures nested in the data tree"""

    def store(obj: Any) -> Any:
        if isinstance(obj, Future):
            futures.append(obj)
        return obj

    futures = []
    traverse(obj, store)
    return futures


@python_app(executors=['default_threads'])
def resolve_futures(obj: Any, inputs: Sequence = ()) -> Any:
    """Replace every nested future with its result. This is blocking, so make sure they are finished."""

    def resolve(obj: Any) -> Any:
        if isinstance(obj, Future):
            return obj.result()
        return copy.copy(obj)  # avoid weird side effects

    if not inputs:
        return obj  # quickly return if there are no futures
    return traverse(obj, resolve)


def resolve_nested_futures(obj: Any) -> AppFuture:
    """Resolve every nested future with its result for (mostly) arbitrary objects.
    Essentially performs list[Future | Any] -> Future[list[Any]] for complex datastructures.
    """
    futures = extract_futures(obj)
    return resolve_futures(obj, inputs=futures)
