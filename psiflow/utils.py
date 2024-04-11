from __future__ import annotations  # necessary for type-guarding class methods

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import typeguard
from ase import Atoms
from ase.data import atomic_numbers
from parsl.app.app import join_app, python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def set_logger(  # hacky
    level: Union[str, int],  # 'DEBUG' or logging.DEBUG
):
    formatter = logging.Formatter(fmt="%(levelname)s - %(name)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    names = [
        "psiflow.data",
        "psiflow.committee",
        "psiflow.metrics",
        "psiflow.execution",
        "psiflow.state",
        "psiflow.learning",
        "psiflow.learning_utils",
        "psiflow.utils",
        "psiflow.parsl_utils",
        "psiflow.models.model",
        "psiflow.models.mace",
        "psiflow.reference._cp2k",
        "psiflow.reference._emt",
        "psiflow.reference._pyscf",
    ]
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)


@typeguard.typechecked
def _sum_integers(a: int, b: int) -> int:
    return a + b


sum_integers = python_app(_sum_integers, executors=["default_threads"])


@typeguard.typechecked
def _combine_futures(inputs: List[Any]) -> List[Any]:
    return list(inputs)


combine_futures = python_app(_combine_futures, executors=["default_threads"])


@typeguard.typechecked
def get_index_element_mask(
    numbers: np.ndarray,
    elements: Optional[List[str]],
    atom_indices: Optional[List[int]],
) -> np.ndarray:
    mask = np.array([True] * len(numbers))

    if elements is not None:
        numbers_to_include = [atomic_numbers[e] for e in elements]
        mask_elements = np.array([False] * len(numbers))
        for number in numbers_to_include:
            mask_elements = np.logical_or(mask_elements, (numbers == number))
        mask = np.logical_and(mask, mask_elements)

    if atom_indices is not None:
        mask_indices = np.array([False] * len(numbers))
        mask_indices[np.array(atom_indices)] = True
        mask = np.logical_and(mask, mask_indices)
    return mask


@typeguard.typechecked
def _dump_json(
    inputs: list = [],
    outputs: list = [],
    **kwargs,
) -> None:
    import json

    import numpy as np

    def convert_to_list(array):
        if not type(array) is np.ndarray:
            return array
        as_list = []
        for item in array:
            as_list.append(convert_to_list(item))
        return as_list

    for name in list(kwargs.keys()):
        value = kwargs[name]
        if type(value) is np.ndarray:
            value = convert_to_list(value)
        kwargs[name] = value
    with open(outputs[0], "w") as f:
        f.write(json.dumps(kwargs))


dump_json = python_app(_dump_json, executors=["default_threads"])


@typeguard.typechecked
def _copy_data_future(
    pass_on_exist: bool = False,
    inputs: List[File] = [],
    outputs: List[File] = [],
) -> None:
    import shutil
    from pathlib import Path

    assert len(inputs) == 1
    assert len(outputs) == 1
    if Path(outputs[1]).is_file() and pass_on_exist:
        return 0
    if Path(inputs[0]).is_file():
        shutil.copyfile(inputs[0], outputs[0])
    else:  # no need to copy empty file
        pass


copy_data_future = python_app(_copy_data_future, executors=["default_threads"])


@typeguard.typechecked
def _copy_app_future(future: Any, inputs: list = [], outputs: list = []) -> Any:
    # inputs/outputs to enforce additional dependencies
    from copy import deepcopy

    return deepcopy(future)


copy_app_future = python_app(_copy_app_future, executors=["default_threads"])


@join_app
@typeguard.typechecked
def log_message(message, *futures):
    logger.info(message.format(*futures))
    return copy_app_future(0)


def _pack(*args):
    return args


pack = python_app(_pack, executors=["default_threads"])


@typeguard.typechecked
def _unpack_i(result: Union[np.ndarray, List, Tuple], i: int) -> Any:
    assert i <= len(result)
    return result[i]


unpack_i = python_app(_unpack_i, executors=["default_threads"])


@typeguard.typechecked
def _save_yaml(
    input_dict: Dict,
    outputs: List[File] = [],
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

    for key, value in extra_keys.items():
        assert key not in input_dict
        input_dict[key] = value
    input_dict = _make_dict_safe(input_dict)
    with open(outputs[0], "w") as f:
        yaml.dump(input_dict, f, default_flow_style=False)


save_yaml = python_app(_save_yaml, executors=["default_threads"])


@typeguard.typechecked
def _save_xml(
    element: ET.Element,
    outputs: list = [],
) -> None:
    import xml.etree.ElementTree as ET_

    with open(outputs[0], "wb") as f:
        f.write(ET_.tostring(element))


save_xml = python_app(_save_xml, executors=["default_threads"])


@typeguard.typechecked
def _read_yaml(inputs: List[File] = [], outputs: List[File] = []) -> dict:
    import yaml

    with open(inputs[0], "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


read_yaml = python_app(_read_yaml, executors=["default_threads"])


@typeguard.typechecked
def _save_txt(data: str, outputs: List[File] = []) -> None:
    with open(outputs[0], "w") as f:
        f.write(data)


save_txt = python_app(_save_txt, executors=["default_threads"])


@typeguard.typechecked
def _app_train_valid_indices(
    effective_nstates: int,
    train_valid_split: float,
) -> Tuple[List[int], List[int]]:
    import numpy as np

    ntrain = int(np.floor(effective_nstates * train_valid_split))
    nvalid = effective_nstates - ntrain
    assert ntrain > 0
    assert nvalid > 0
    order = np.arange(ntrain + nvalid, dtype=int)
    np.random.shuffle(order)
    train_list = list(order[:ntrain])
    valid_list = list(order[ntrain : (ntrain + nvalid)])
    return [int(i) for i in train_list], [int(i) for i in valid_list]


app_train_valid_indices = python_app(
    _app_train_valid_indices, executors=["default_threads"]
)


@typeguard.typechecked
def get_train_valid_indices(
    effective_nstates: AppFuture,
    train_valid_split: float,
) -> Tuple[AppFuture, AppFuture]:
    future = app_train_valid_indices(effective_nstates, train_valid_split)
    return unpack_i(future, 0), unpack_i(future, 1)


@typeguard.typechecked
def resolve_and_check(path: Path) -> Path:
    path = path.resolve()
    if Path.cwd() in path.parents:
        pass
    elif path.exists() and Path.cwd().samefile(path):
        pass
    else:
        raise ValueError(
            "requested file and/or path at location: {}"
            "\nwhich is not in the present working directory: {}"
            "\npsiflow can only load and/or save in its present "
            "working directory because this is the only directory"
            " that will get bound into the container.".format(path, Path.cwd())
        )
    return path


def compute_error(
    atoms_0: Atoms,
    atoms_1: Atoms,
    metric: str,
    mask: np.ndarray,
    properties: List[str],
) -> tuple:
    import numpy as np
    from ase.units import Pascal

    errors = np.zeros(len(properties))
    assert mask.dtype == bool
    if not np.any(mask):  # should still return a tuple
        return tuple([np.nan for i in range(len(properties))])
    if "energy" in properties:
        assert "energy" in atoms_0.info.keys()
        assert "energy" in atoms_1.info.keys()
    if "forces" in properties:
        assert "forces" in atoms_0.arrays.keys()
        assert "forces" in atoms_1.arrays.keys()
    if "stress" in properties:
        assert "stress" in atoms_0.info.keys()
        assert "stress" in atoms_1.info.keys()
    for j, property_ in enumerate(properties):
        if property_ == "energy":
            array_0 = np.array([atoms_0.info["energy"]]).reshape((1, 1))
            array_1 = np.array([atoms_1.info["energy"]]).reshape((1, 1))
            array_0 /= len(atoms_0)  # per atom energy error
            array_1 /= len(atoms_1)
            array_0 *= 1000  # in meV/atom
            array_1 *= 1000
        elif property_ == "forces":
            array_0 = atoms_0.arrays["forces"][mask, :]
            array_1 = atoms_1.arrays["forces"][mask, :]
            array_0 *= 1000  # in meV/angstrom
            array_1 *= 1000
        elif property_ == "stress":
            array_0 = atoms_0.info["stress"].reshape((1, 9))
            array_1 = atoms_1.info["stress"].reshape((1, 9))
            array_0 /= 1e6 * Pascal  # in MPa
            array_1 /= 1e6 * Pascal
        else:
            raise ValueError("property {} unknown!".format(property_))
        if metric == "mae":
            errors[j] = np.mean(np.abs(array_0 - array_1))
        elif metric == "rmse":
            errors[j] = np.sqrt(np.mean((array_0 - array_1) ** 2))
        elif metric == "max":
            errors[j] = np.max(np.linalg.norm(array_0 - array_1, axis=1))
        else:
            raise ValueError("metric {} unknown!".format(metric))
    return tuple([float(e) for e in errors])


def is_lower_triangular(rvecs):
    """Returns whether rvecs are in lower triangular form

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (
        rvecs[0, 0] > 0
        and rvecs[1, 1] > 0  # positive volumes
        and rvecs[2, 2] > 0
        and rvecs[0, 1] == 0
        and rvecs[0, 2] == 0  # lower triangular
        and rvecs[1, 2] == 0
    )


def is_reduced(rvecs):
    """Returns whether rvecs are in reduced form

    OpenMM puts requirements on the components of the box vectors.
    Essentially, rvecs has to be a lower triangular positive definite matrix
    where additionally (a_x > 2*|b_x|), (a_x > 2*|c_x|), and (b_y > 2*|c_y|).

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (
        rvecs[0, 0] > abs(2 * rvecs[1, 0])
        and rvecs[0, 0] > abs(2 * rvecs[2, 0])  # b mostly along y axis
        and rvecs[1, 1] > abs(2 * rvecs[2, 1])  # c mostly along z axis
        and is_lower_triangular(rvecs)  # c mostly along z axis
    )


def transform_lower_triangular(pos, rvecs, reorder=False):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    reorder : bool
        whether box vectors are reordered from largest to smallest

    """
    if reorder:  # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(rvecs, axis=1)
        ordering = np.argsort(norms)[::-1]  # largest first
        a = rvecs[ordering[0], :].copy()
        b = rvecs[ordering[1], :].copy()
        c = rvecs[ordering[2], :].copy()
        rvecs[0, :] = a[:]
        rvecs[1, :] = b[:]
        rvecs[2, :] = c[:]
    q, r = np.linalg.qr(rvecs.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r))  # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors  # full (improper) rotation
    pos[:] = pos @ rotation
    rvecs[:] = rvecs @ rotation
    assert np.allclose(rvecs, np.linalg.cholesky(rvecs @ rvecs.T))
    rvecs[0, 1] = 0
    rvecs[0, 2] = 0
    rvecs[1, 2] = 0


def reduce_box_vectors(rvecs):
    """Uses linear combinations of box vectors to obtain the reduced form

    The reduced form of a cell matrix is lower triangular, with additional
    constraints that enforce vector b to lie mostly along the y-axis and vector
    c to lie mostly along the z axis.

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows. These should already by in
        lower triangular form.

    """
    # simple reduction algorithm only works on lower triangular cell matrices
    assert is_lower_triangular(rvecs)
    # replace c and b with shortest possible vectors to ensure
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    rvecs[2, :] = rvecs[2, :] - rvecs[1, :] * np.round(rvecs[2, 1] / rvecs[1, 1])
    rvecs[2, :] = rvecs[2, :] - rvecs[0, :] * np.round(rvecs[2, 0] / rvecs[0, 0])
    rvecs[1, :] = rvecs[1, :] - rvecs[0, :] * np.round(rvecs[1, 0] / rvecs[0, 0])


@typeguard.typechecked
def _check_distances(state: Atoms, threshold: float):
    import numpy as np
    from ase.geometry.geometry import find_mic

    from psiflow.data import NullState

    if state == NullState:
        return NullState
    nrows = int(len(state) * (len(state) - 1) / 2)
    deltas = np.zeros((nrows, 3))
    count = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            deltas[count] = state.positions[i] - state.positions[j]
            count += 1
    assert count == nrows
    if state.pbc.all():
        deltas, _ = find_mic(deltas, state.cell)
    check = np.all(np.linalg.norm(deltas, axis=1) > threshold)
    if check:
        return state
    else:
        return NullState


check_distances = python_app(_check_distances, executors=["default_threads"])


@typeguard.typechecked
def apply_temperature_ramp(
    T_min: float, T_max: float, nsteps: int, current_temperature: float
) -> float:
    assert T_max > T_min
    if nsteps > 1:
        delta_beta = (1 / T_min - 1 / T_max) / (nsteps - 1)
        next_beta = 1 / current_temperature - delta_beta
        if (next_beta > 0) and (next_beta > 1 / T_max):
            return 1 / next_beta
        else:
            return T_max
    else:
        return T_max
