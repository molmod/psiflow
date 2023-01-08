from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Tuple, Dict
import typeguard
import os
import sys
import tempfile
import numpy as np
import wandb
import importlib
from pathlib import Path

from ase.data import atomic_numbers

from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.config import Config


@typeguard.typechecked
def get_parsl_config_from_file(
        path_config: Union[Path, str],
        path_internal: Union[Path, str],
        ) -> Config:
    path_config = Path(path_config)
    assert path_config.is_file()
    # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    spec = importlib.util.spec_from_file_location('module.name', path_config)
    parsl_config_module = importlib.util.module_from_spec(spec)
    sys.modules['module.name'] = parsl_config_module
    spec.loader.exec_module(parsl_config_module)
    return parsl_config_module.get_config(path_internal)


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
def _new_file(path: Union[Path, str], prefix: str, suffix: str) -> str:
    _, name = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=path,
            )
    return name


@typeguard.typechecked
def _copy_data_future(inputs: List[File] = [], outputs: List[File] = []) -> None:
    import shutil
    assert len(inputs)  == 1
    assert len(outputs) == 1
    shutil.copyfile(inputs[0], outputs[0])
copy_data_future = python_app(_copy_data_future, executors=['default'])


@typeguard.typechecked
def _copy_app_future(future: Any) -> Any:
    from copy import deepcopy
    return deepcopy(future)
copy_app_future = python_app(_copy_app_future, executors=['default'])


@typeguard.typechecked
def _unpack_i(result: Union[List, Tuple], i: int) -> Any:
    return result[i]
unpack_i = python_app(_unpack_i, executors=['default'])


@typeguard.typechecked
def _save_yaml(input_dict: Dict, outputs: List[File] = []) -> None:
    import yaml
    with open(outputs[0], 'w') as f:
        yaml.dump(input_dict, f, default_flow_style=False)
save_yaml = python_app(_save_yaml, executors=['default'])


@typeguard.typechecked
def _save_txt(data: str, outputs: List[File] = []) -> None:
    with open(outputs[0], 'w') as f:
        f.write(data)
save_txt = python_app(_save_txt, executors=['default'])

@typeguard.typechecked
def _log_data_to_wandb(
        run_name: str,
        group: str,
        project: str,
        error_x_axis: str,
        names: List[str],
        inputs: List[List[List]] = [], # list of 2D tables
        ) -> None:
    from pathlib import Path
    import tempfile
    import wandb
    path_wandb = Path(tempfile.mkdtemp())
    wandb.init(
            name=run_name,
            group=group,
            project=project,
            resume='allow',
            dir=path_wandb,
            )
    wandb_log = {}
    assert len(names) == len(inputs)
    for name, data in zip(names, inputs):
        table = wandb.Table(columns=data[0], data=data[1:])
        if name in ['training', 'validation', 'failed']:
            errors_to_plot = [] # check which error labels are present
            for l in data[0]:
                if l.endswith('energy') or l.endswith('forces') or l.endswith('stress'):
                    errors_to_plot.append(l)
            assert error_x_axis in data[0]
            for error in errors_to_plot:
                title = name + '_' + error
                wandb_log[title] = wandb.plot.scatter(
                        table,
                        error_x_axis,
                        error,
                        title=title,
                        )
        else:
            wandb_log[name + '_table'] = table
    assert path_wandb.is_dir()
    os.environ['WANDB_SILENT'] = 'True' # suppress logs
    wandb.log(wandb_log)
    wandb.finish()
log_data_to_wandb = python_app(_log_data_to_wandb, executors=['default'])


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
    return list(range(ntrain)), list(range(ntrain, ntrain + nvalid))
app_train_valid_indices = python_app(_app_train_valid_indices)


def get_train_valid_indices(
        effective_nstates: AppFuture,
        train_valid_split: float,
        ) -> Union[AppFuture, AppFuture]:
    future = app_train_valid_indices(effective_nstates, train_valid_split)
    return unpack_i(future, 0), unpack_i(future, 1)
