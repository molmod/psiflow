from __future__ import annotations  # necessary for type-guarding class methods

import inspect
from pathlib import Path
from typing import Optional, Union, get_args, get_origin, get_type_hints

import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.geometry import Geometry
from psiflow.utils import copy_data_future, resolve_and_check

_DataFuture = Union[File, DataFuture]


class Serializable:
    pass


def dummy(*args, **kwargs):
    return None


def create_getter(name, kind, type_hint):
    @typeguard.typechecked
    def getter(self) -> type_hint:
        return getattr(self, "_{}".format(kind))[name]

    return getter


def create_setter(name, kind, type_hint):
    @typeguard.typechecked
    def setter(self, value: type_hint) -> None:
        _dict = getattr(self, "_{}".format(kind))
        _dict[name] = value

    return setter


def update_init(init_func):
    def wrapper(self, *args, **kwargs):
        self._geoms = {}
        self._files = {}
        self._attrs = {}
        self._serial = {}
        return init_func(self, *args, **kwargs)

    return wrapper


def serializable(cls):
    """decorator to make class serializable"""
    class_dict = dict(cls.__dict__)
    for name, type_hint in get_type_hints(cls).items():
        if get_origin(type_hint) in [Union, Optional, list, tuple]:
            args = get_args(type_hint)
            if (File in args) or (DataFuture in args):
                kind = "files"
            else:
                if Geometry in args:
                    kind = "geoms"
                else:
                    kind = "attrs"
                for arg in args:
                    if inspect.isclass(arg):
                        if issubclass(arg, Serializable):  # weird
                            kind = "serial"
        else:
            if not inspect.isclass(type_hint):
                raise ValueError(
                    "{} is formally not a class ({})".format(type_hint, name)
                )
            if issubclass(type_hint, Serializable):
                kind = "serial"
            elif type_hint is Geometry:
                kind = "geoms"
            else:
                kind = "attrs"
        getter = create_getter(name, kind, type_hint)
        setter = create_setter(name, kind, type_hint)
        class_dict[name] = property(getter, setter)

    if "__init__" not in class_dict:
        class_dict["__init__"] = dummy
    class_dict["__init__"] = update_init(
        class_dict["__init__"]
    )  # create _attrs / _files / _serial

    bases = cls.__mro__
    if bases is not None:
        if Serializable in bases:
            pass
        else:
            bases = (Serializable,) + bases
    else:
        bases = (Serializable,)
    new_cls = type(
        cls.__name__,
        bases,
        class_dict,
    )
    return new_cls


@typeguard.typechecked
def _dump_json(
    inputs: list = [],
    outputs: list = [],
    **kwargs,
) -> dict:
    import json

    import numpy as np
    from parsl.dataflow.futures import AppFuture

    def convert_to_list(array):
        if not type(array) is np.ndarray:
            return array
        as_list = []
        for item in array:
            as_list.append(convert_to_list(item))
        return as_list

    def descend_and_wait(value):
        if type(value) in [AppFuture, DataFuture]:
            value = value.result()
        if type(value) is dict:
            for key in list(value.keys()):
                value[key] = descend_and_wait(value[key])
        if type(value) in [list]:  # do not allow futures in tuples!
            for i in range(len(value)):
                value[i] = descend_and_wait(value[i])
        return value

    # descend_and_wait(kwargs)
    # print(kwargs)
    # for name, value in kwargs.items():
    #    print(name, type(value))

    kwargs = descend_and_wait(kwargs)

    for name in list(kwargs.keys()):
        value = kwargs[name]
        if type(value) is np.ndarray:
            value = convert_to_list(value)
        kwargs[name] = value

    s = json.dumps(kwargs)
    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write(s)
    return kwargs


dump_json = python_app(_dump_json, executors=["default_threads"])


@typeguard.typechecked
def serialize(
    obj: Serializable,
    path_json: Optional[Path] = None,
    copy_to: Optional[Path] = None,
) -> AppFuture:
    if path_json is not None:
        path_json = resolve_and_check(path_json)
    data = {
        "_attrs": dict(obj._attrs),
    }

    # dump_json waits for all futures in this list
    inputs = (
        list(obj._attrs.values())
        + list(obj._files.values())
        # + list(obj._serial.values())
        # + list(obj._geoms.values())
    )

    # populate _files dict;
    # if data futures need to be copied, this adds the copy operation to inputs
    _files = {}
    if copy_to is None:
        for key, _file in obj._files.items():
            _files[key] = _file.filepath
    else:
        copy_to.mkdir(exist_ok=True)
        for key, _file in obj._files.items():
            new_path = copy_to / Path(_file.filepath).name
            new_file = copy_data_future(
                pass_on_exist=True,  # e.g. identical hamiltonians in different walkers
                inputs=[_file],
                outputs=[File(new_path)],
            ).outputs[0]
            _files[key] = new_file.filepath
            inputs.append(new_file)
    data["_files"] = _files

    # populate _serial dict;
    # adds result of sub serialize calls to inputs
    _serial = {}
    for name, serial in obj._serial.items():
        if serial is None:  # optional types
            _serial[name] = None
            continue
        if type(serial) in [list, tuple]:
            serialized = [serialize(s, path_json=None, copy_to=copy_to) for s in serial]
            inputs += serialized
        else:
            serialized = serialize(serial, path_json=None, copy_to=copy_to)
            inputs.append(serialized)
        _serial[name] = serialized
    data["_serial"] = _serial

    # populate _geoms dict:
    # generate Geometry and AppFuture[Geometry] strings
    @python_app(executors=["default_threads"])
    def to_string(geometry: Optional[Geometry]) -> str:
        if geometry is None:
            return ""
        else:
            return geometry.to_string()

    _geoms = {}
    for key, value in obj._geoms.items():
        _geoms[key] = to_string(value)
    data["_geoms"] = _geoms
    inputs += list(_geoms.values())

    if path_json is not None:
        outputs = [File(str(path_json))]
    else:
        outputs = []

    return dump_json(
        **{obj.__class__.__name__: data},
        inputs=inputs,
        outputs=outputs,
    )


@typeguard.typechecked
def deserialize(data: dict, custom_cls: Optional[list] = None):
    from psiflow.data import Dataset
    from psiflow.hamiltonians import EinsteinCrystal, MACEHamiltonian, PlumedHamiltonian
    from psiflow.models import MACE
    from psiflow.reference import CP2K
    from psiflow.sampling import (
        Metadynamics,
        OrderParameter,
        ReplicaExchange,
        SimulationOutput,
        Walker,
    )

    SERIALIZABLES = {}
    if custom_cls is None:
        custom_cls = []
    for cls in custom_cls + [
        Dataset,
        MACE,
        CP2K,
        MACEHamiltonian,
        EinsteinCrystal,
        PlumedHamiltonian,
        Metadynamics,
        OrderParameter,
        ReplicaExchange,
        SimulationOutput,
        Walker,
    ]:
        SERIALIZABLES[cls.__name__] = cls

    cls_name = list(data.keys())[0]
    cls = SERIALIZABLES.get(cls_name, None)
    assert cls is not None

    obj = cls.__new__(cls)
    obj._files = {k: File(v) for k, v in data[cls_name]["_files"].items()}
    obj._attrs = data[cls_name]["_attrs"]
    obj._geoms = {
        k: Geometry.from_string(s, natoms=None)
        for k, s in data[cls_name]["_geoms"].items()
    }
    _serial = {}
    for key, value in data[cls_name]["_serial"].items():
        if value is None:
            _serial[key] = value
        elif type(value) in [list, tuple]:
            _serial[key] = [deserialize(v, custom_cls=custom_cls) for v in value]
        else:
            _serial[key] = deserialize(value, custom_cls=custom_cls)
    obj._serial = _serial
    if hasattr(obj, "_create_apps"):
        obj._create_apps()
    return obj
