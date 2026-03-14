import shutil
import json
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Sequence

from parsl import File, python_app
from parsl.dataflow.futures import AppFuture, DataFuture

import psiflow
from psiflow.geometry import Geometry
from psiflow.utils.future import resolve_nested_futures


# TODO: verify which attributes need to be serialized with a _to_serialize key?


CLS_KEY = "PSIFLOW_CLS"
SKIP_INIT = "SKIP_INIT"
SERIALIZABLE_CLS = {}

_DataFuture = Union[File, DataFuture]  # TODO: does not belong here


# TODO: temporary patch
def serializable(obj: Any) -> Any:
    return obj


class SerializationError(TypeError):
    pass


def register_serializable(cls):
    SERIALIZABLE_CLS[cls.__name__] = cls
    return cls


def serialize(
    obj: Any, path_json: Optional[Path] = None, copy_to: Optional[Path] = None
) -> AppFuture:
    """JSON serialization for psiflow classes"""
    outputs = []
    if path_json is not None:
        outputs = [File(psiflow.resolve_and_check(path_json))]
    future = resolve_nested_futures(obj)
    return serialize_object(future, copy_to, outputs=outputs)


def _deserialize(json_str: str) -> Any:
    """Reconstruct a psiflow object"""

    return json.loads(json_str, object_hook=deserialize_hook)


deserialize = python_app(_deserialize, executors=["default_threads"])


def _serialize_object(
    obj: Any,
    copy_to: Optional[Path],
    outputs: Sequence[File],
) -> str:
    """Serialize a psiflow object. It should not contain any futures."""
    try:
        json_str = json.dumps(obj, cls=JSONEncoder, copy_to=copy_to)
    except TypeError as e:
        cls_set = set(SERIALIZABLE_CLS.keys()) or {}
        msg = f"Failed to serialize with error '{e}'. Is it an instance of {cls_set}?"
        raise SerializationError(msg)
    if outputs:
        with open(outputs[0], "w") as f:
            f.write(json_str)
    return json_str


serialize_object = python_app(_serialize_object, executors=["default_threads"])


class JSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self.copy_to = copy_to = kwargs.pop("copy_to", None)
        if copy_to is not None:
            copy_to.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def default(self, obj: Any) -> dict[str, Any]:
        """How to handle special class instances"""
        name = obj.__class__.__name__
        match obj:
            case File():
                return self._handle_file(obj)
            case Geometry():
                return {CLS_KEY: "Geometry", "data": obj.to_string()}
            case _ if name in SERIALIZABLE_CLS:  # class instances
                return {CLS_KEY: name} | vars(obj)
            case _ if obj in SERIALIZABLE_CLS.values():  # classes
                return {CLS_KEY: obj.__name__} | {SKIP_INIT: True}
            case _:
                return super().default(obj)  # fall back to default behaviour

    def _handle_file(self, obj: File) -> dict[str, str]:
        if self.copy_to is None:
            path = obj.filepath
        else:
            path = self.copy_to / obj.filename
            if path.is_file():
                pass  # e.g. identical hamiltonians in different walkers
            else:
                shutil.copy(obj.filepath, path)
        return {CLS_KEY: "File", "path": str(path)}


def deserialize_hook(data: dict) -> Any:
    """Reconstruct psiflow objects. Let JSON handle the rest."""
    if CLS_KEY not in data:
        return data
    cls_name = data.pop(CLS_KEY)

    if cls_name == "File":
        return File(Path(data["path"]))
    elif cls_name == "Geometry":
        return Geometry.from_string(data["data"])
    elif cls_name not in SERIALIZABLE_CLS:
        cls_set = set(SERIALIZABLE_CLS.keys()) or {}
        msg = f"Custom class '{cls_name}' not in {cls_set}. Cannot deserialize.."
        raise TypeError(msg)

    cls = SERIALIZABLE_CLS.get(cls_name)
    if data.get(SKIP_INIT):
        return cls  # return class instead of instance

    obj = cls.__new__(cls)
    for k, v in data.items():
        setattr(obj, k, v)
    return obj
