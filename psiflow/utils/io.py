import json
import xml.etree.ElementTree as ET
from typing import Any
from collections.abc import Sequence

import numpy as np
from parsl.app.app import python_app
from parsl.data_provider.files import File

from psiflow.serialization import JSONEncoder

# TODO: check which of these methods is actively used


def _save_yaml(
    input_dict: dict,
    outputs: list[File] = [],
    **extra_keys: Any,
) -> None:
    import yaml

    def _make_dict_safe(arg):
        # walks through dict and converts numpy types to python natives
        for key in list(arg.keys()):
            if hasattr(arg[key], "item"):
                arg[key] = arg[key].item()
            elif type(arg[key]) is dict:
                arg[key] = _make_dict_safe(arg[key])
            else:
                pass
        return arg

    input_dict = dict(input_dict)
    for key, value in extra_keys.items():
        assert key not in input_dict
        input_dict[key] = value
    input_dict = _make_dict_safe(input_dict)
    with open(outputs[0], "w") as f:
        yaml.dump(input_dict, f, default_flow_style=False)


save_yaml = python_app(_save_yaml, executors=["default_threads"])


def _save_xml(
    element: ET.Element,
    outputs: list = [],
) -> None:
    tree = ET.ElementTree(element)
    ET.indent(tree, "  ")
    tree.write(outputs[0], encoding="utf-8", xml_declaration=True)


save_xml = python_app(_save_xml, executors=["default_threads"])


def _load_numpy(inputs: list[File] = [], **kwargs) -> np.ndarray:
    return np.loadtxt(inputs[0], **kwargs)


load_numpy = python_app(_load_numpy, executors=["default_threads"])


def _read_yaml(inputs: list[File] = []) -> dict:
    import yaml

    with open(inputs[0], "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


read_yaml = python_app(_read_yaml, executors=["default_threads"])


def _save_txt(data: str, outputs: list[File] = []) -> None:
    with open(outputs[0], "w") as f:
        f.write(data)


save_txt = python_app(_save_txt, executors=["default_threads"])


def _load_metrics(inputs: list = []) -> np.recarray:
    # TODO: stop using recarrays
    return np.load(inputs[0], allow_pickle=True)


load_metrics = python_app(_load_metrics, executors=["default_threads"])


def _save_metrics(data: np.recarray, outputs: list = []) -> None:
    # TODO: stop using recarrays
    with open(outputs[0], "wb") as f:
        data.dump(f)


save_metrics = python_app(_save_metrics, executors=["default_threads"])


def _dump_json(
    inputs: Sequence = (),
    outputs: Sequence = (),
    **kwargs,
) -> str:
    assert len(outputs) <= 1
    json_str = json.dumps(kwargs, cls=JSONEncoder)
    if len(outputs) == 0:
        return json_str
    with open(outputs[0], "w") as f:
        f.write(json_str)
    return json_str



dump_json = python_app(_dump_json, executors=["default_threads"])


def _save_npz(
    data: dict[str, np.ndarray], outputs: list[File], **kwargs: np.ndarray
) -> None:
    """"""
    assert len(outputs) == 1
    np.savez(outputs[0].filepath, **(data | kwargs))


save_npz = python_app(_save_npz, executors=["default_threads"])


def _load_npz(file_in: File) -> dict[str, Any]:
    """"""
    data: dict[str, np.ndarray] = dict(np.load(file_in))
    for k, v in data.items():
        if v.size == 1:
            data[k] = v.item()  # convert to native python type
    return data


load_npz = python_app(_load_npz, executors=["default_threads"])
