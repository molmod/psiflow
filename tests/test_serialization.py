from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import pytest
from parsl import File
from parsl.dataflow.futures import Future

import psiflow
from psiflow.geometry import Geometry
from psiflow.serialization import (
    SERIALIZABLE_CLS,
    register_serializable,
    SerializationError,
)
from psiflow.utils.future import resolve_nested_futures


@register_serializable
@dataclass
class Custom1:
    foo: Any
    bar: list[Any]
    baz: dict[str, Any]
    bam: Future
    boa: Optional[Geometry] = None


@register_serializable
class Custom2(Custom1):
    pass  # just to mix classes


@dataclass
class Custom3:  # not marked as serializable
    foo: Any


def wrap_in_future(obj: Any) -> Future[Any]:
    future = Future()
    future.set_result(obj)
    return future


def test_resolve_futures():
    """Verify that resolve_nested_futures works as expected."""
    instance = Custom1(
        foo=Custom3(wrap_in_future(42)),
        bar=[42, None, File(""), wrap_in_future(42)],
        baz={"1": "blankets", "2": None, "3": File(""), "4": wrap_in_future(42)},
        bam=wrap_in_future(Custom2),
    )
    out = resolve_nested_futures(instance).result()

    # check for side effects
    futures = [instance.bam, instance.bar[-1], instance.baz["4"], instance.foo.foo]
    assert all(isinstance(f, Future) for f in futures)
    assert isinstance(instance.bam, Future)
    assert isinstance(instance.bar[-1], Future)
    assert isinstance(instance.baz["4"], Future)
    assert isinstance(instance.foo.foo, Future)
    assert instance.bar[0] == 42 and instance.baz["1"] == "blankets"

    # check resolved futures
    assert out.bam == Custom2
    assert out.bar[-1] == 42
    assert out.baz["4"] == 42
    assert out.foo.foo == 42
    assert out.bar[0] == 42 and out.baz["1"] == "blankets"


def test_serial_simple(tmp_path):
    """"""
    assert "Custom1" in SERIALIZABLE_CLS
    assert "Custom2" in SERIALIZABLE_CLS
    assert "Custom3" not in SERIALIZABLE_CLS

    file1 = File(tmp_path / "1.log")
    file2 = File(tmp_path / "2.log")
    Path(file1.filepath).touch()
    Path(file2.filepath).touch()
    instance = Custom1(
        foo=Custom3(wrap_in_future(42)),
        bar=[42, None, file1, wrap_in_future(42)],
        baz={"1": "blankets", "2": None, "3": file2, "4": wrap_in_future(42)},
        bam=wrap_in_future(Custom2),
    )
    with pytest.raises(SerializationError):  # cannot do Custom3
        psiflow.serialize(instance).result()
    SERIALIZABLE_CLS[Custom3.__name__] = Custom3  # can do Custom3

    # test basic serialization
    future = psiflow.serialize(instance)
    json_str = future.result()
    assert isinstance(json_str, str)
    instance1 = psiflow.deserialize(future).result()
    instance2 = resolve_nested_futures(instance).result()
    assert type(instance1) == type(instance2)
    assert instance1.foo == instance2.foo
    assert instance1.bam == instance2.bam
    file1, file2 = instance1.bar.pop(2), instance2.bar.pop(2)
    assert file1.filepath == file2.filepath
    assert instance1.bar == instance2.bar
    file1, file2 = instance1.baz.pop("3"), instance2.baz.pop("3")
    assert file1.filepath == file2.filepath
    assert instance1.baz == instance2.baz

    # test copy_to serialization
    copy_dir = tmp_path / "dir"
    psiflow.serialize(instance, copy_to=copy_dir).result()
    assert copy_dir.is_dir()
    assert (copy_dir / "1.log").is_file()
    assert (copy_dir / "2.log").is_file()
