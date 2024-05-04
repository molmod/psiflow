import json
import os
from pathlib import Path
from typing import Optional, Union

import pytest
import typeguard
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, NullState, new_nullstate
from psiflow.utils import copy_app_future


def test_serial_simple(tmp_path):
    @psiflow.serializable
    class SomeSerial:
        pass

    @typeguard.typechecked
    class Test:
        foo: int
        bar: psiflow._DataFuture
        baz: Union[float, str]
        bam: Optional[SomeSerial]
        bao: SomeSerial
        bap: list[SomeSerial, ...]
        baq: Union[Geometry, AppFuture]
        bas: Geometry

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    new_cls = psiflow.serializable(Test)
    instance = new_cls(
        foo=3,
        bar=File("asdfl"),
        baz="asdflk",
        bam=None,
        bao=SomeSerial(),
        bap=[SomeSerial(), SomeSerial()],
        baq=copy_app_future(NullState),
        bas=new_nullstate(),
    )
    assert instance.foo == 3
    assert instance._attrs["foo"] == 3

    # test independence
    instance._attrs["test"] = 1
    instance_ = new_cls(foo=4, bar=File("asdfl"))
    assert "test" not in instance_._attrs
    assert instance_.foo == 4
    assert instance.foo == 3

    assert tuple(instance._files.keys()) == ("bar",)
    assert tuple(instance._attrs.keys()) == ("foo", "baz", "test")
    assert tuple(instance._serial.keys()) == ("bam", "bao", "bap")
    assert type(instance._serial["bap"]) is list
    assert len(instance._serial["bap"]) == 2
    assert len(instance._geoms) == 2
    assert "baq" in instance._geoms
    assert "bas" in instance._geoms

    # serialization/deserialization of 'complex' Test instance
    json_dump = psiflow.serialize(instance).result()
    instance_ = psiflow.deserialize(json_dump, custom_cls=[new_cls, SomeSerial])

    assert instance.foo == instance_.foo
    assert instance.bar.filepath == instance_.bar.filepath
    assert instance.baz == instance_.baz
    assert instance.bam == instance_.bam
    assert type(instance_.bao) is SomeSerial
    assert len(instance_.bap) == 2
    assert type(instance_.bap[0]) is SomeSerial
    assert type(instance_.bap[1]) is SomeSerial
    assert id(instance) != id(instance_)
    assert isinstance(instance_.baq, Geometry)
    assert instance_.baq == NullState
    assert instance_.bas == NullState

    # check classes created before test execution, e.g. Dataset
    data = Dataset([NullState])
    assert "extxyz" in data._files
    assert len(data._attrs) == 0
    assert len(data._serial) == 0
    with pytest.raises(typeguard.TypeCheckError):  # try something stupid
        data.extxyz = 0

    # test getter / setter
    data.extxyz = File("some_file")
    assert type(data.extxyz) is File

    # test basic serialization
    dumped_json = psiflow.serialize(data).result()
    assert "Dataset" in dumped_json
    data_dict = json.loads(dumped_json)
    assert len(data_dict["Dataset"]["_attrs"]) == 0
    assert len(data_dict["Dataset"]["_serial"]) == 0
    assert len(data_dict["Dataset"]["_files"]) == 1
    assert data_dict["Dataset"]["_files"]["extxyz"] == data.extxyz.filepath

    # test copy_to serialization
    data = Dataset([NullState])
    data.extxyz.result()
    filename = Path(data.extxyz.filepath).name
    assert os.path.exists(data.extxyz.filepath)
    dumped_json = psiflow.serialize(data, copy_to=tmp_path / "test").result()
    os.remove(data.extxyz.filepath)
    assert (tmp_path / "test").exists()
    assert (tmp_path / "test" / filename).exists()  # new file
