from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Tuple, Dict
import typeguard
import os
import sys
import logging
import tempfile
import numpy as np
import importlib
from pathlib import Path

from ase.data import atomic_numbers

from parsl.executors.base import ParslExecutor
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def set_logger( # hacky
        level: Union[str, int], # 'DEBUG' or logging.DEBUG
        ):
    formatter = logging.Formatter(fmt='%(levelname)s - %(name)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    names = [
            'psiflow.data',
            'psiflow.generate',
            'psiflow.execution',
            'psiflow.wandb_utils',
            'psiflow.state',
            'psiflow.learning',
            'psiflow.utils',
            'psiflow.models.base',
            'psiflow.models._mace',
            'psiflow.models._nequip',
            'psiflow.reference._cp2k',
            'psiflow.walkers.bias',
            ]
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)


@typeguard.typechecked
def _sum_integers(a: int, b: int) -> int:
    return a + b
sum_integers = python_app(_sum_integers, executors=['default'])


@typeguard.typechecked
def _create_if_empty(outputs: List[File] = []) -> None:
    try:
        with open(inputs[1], 'r') as f:
            f.read()
    except FileNotFoundError: # create it if it doesn't exist
        with open(inputs[1], 'w+') as f:
            f.write('')
create_if_empty = python_app(_create_if_empty, executors=['default'])


@typeguard.typechecked
def _combine_futures(inputs: List[Any]) -> List[Any]:
    return list(inputs)
combine_futures = python_app(_combine_futures, executors=['default'])


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
def _copy_data_future(inputs: List[File] = [], outputs: List[File] = []) -> None:
    import shutil
    from pathlib import Path
    assert len(inputs)  == 1
    assert len(outputs) == 1
    if Path(inputs[0]).is_file():
        shutil.copyfile(inputs[0], outputs[0])
    else: # no need to copy empty file
        pass
copy_data_future = python_app(_copy_data_future, executors=['default'])


@typeguard.typechecked
def _copy_app_future(future: Any) -> Any:
    from copy import deepcopy
    return deepcopy(future)
copy_app_future = python_app(_copy_app_future, executors=['default'])


@typeguard.typechecked
def _pack(*args):
    return args
pack = python_app(_pack, executors=['default'])


@typeguard.typechecked
def _unpack_i(result: Union[List, Tuple], i: int) -> Any:
    return result[i]
unpack_i = python_app(_unpack_i, executors=['default'])


@typeguard.typechecked
def _save_yaml(input_dict: Dict, outputs: List[File] = []) -> None:
    import yaml
    def _make_dict_safe(arg):
        # walks through dict and converts numpy types to python natives
        for key in list(arg.keys()):
            if hasattr(arg[key], 'item'):
                arg[key] = arg[key].item()
            elif type(arg[key]) == dict:
                arg[key] = _make_dict_safe(arg[key])
            else:
                pass
        return arg
    input_dict = _make_dict_safe(input_dict)
    with open(outputs[0], 'w') as f:
        yaml.dump(input_dict, f, default_flow_style=False)
save_yaml = python_app(_save_yaml, executors=['default'])


@typeguard.typechecked
def _read_yaml(inputs: List[File] = [], outputs: List[File] = []) -> dict:
    import yaml
    with open(inputs[0], 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict
read_yaml = python_app(_read_yaml, executors=['default'])



@typeguard.typechecked
def _save_txt(data: str, outputs: List[File] = []) -> None:
    with open(outputs[0], 'w') as f:
        f.write(data)
save_txt = python_app(_save_txt, executors=['default'])


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
    valid_list = list(order[ntrain:(ntrain + nvalid)])
    return [int(i) for i in train_list], [int(i) for i in valid_list]
app_train_valid_indices = python_app(_app_train_valid_indices, executors=['default'])


@typeguard.typechecked
def get_train_valid_indices(
        effective_nstates: AppFuture,
        train_valid_split: float,
        ) -> Tuple[AppFuture, AppFuture]:
    future = app_train_valid_indices(effective_nstates, train_valid_split)
    return unpack_i(future, 0), unpack_i(future, 1)


@typeguard.typechecked
def get_active_executor(label: str) -> ParslExecutor:
    from parsl.dataflow.dflow import DataFlowKernelLoader
    dfk = DataFlowKernelLoader.dfk()
    config = dfk.config
    for executor in config.executors:
        if executor.label == label:
            return executor
    raise ValueError('executor with label {} not found!'.format(label))


@typeguard.typechecked
def resolve_and_check(path: Path) -> Path:
    path = path.resolve()
    if not Path.cwd() in path.parents:
        raise ValueError('requested file and/or path at location: {}'
                '\nwhich is not in the present working directory: {}'
                '\npsiflow can only load and/or save in its present '
                'working directory because this is the only directory'
                ' that will get bound into the container.'.format(
                    path, Path.cwd()))
    return path
