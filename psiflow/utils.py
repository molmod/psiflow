from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Tuple, Dict
import typeguard
import os
import sys
import tempfile
import numpy as np
import wandb
import importlib
import pkgutil
from pathlib import Path

from ase.data import atomic_numbers

from parsl.executors.base import ParslExecutor
from parsl.app.app import python_app, join_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.config import Config
import parsl.providers.slurm.slurm # to define custom slurm provider

import psiflow

import math # slurm provider imports
import os
import time
import logging
from parsl.providers.base import JobState, JobStatus
from parsl.utils import wtime_to_minutes
from parsl.providers.slurm.template import template_string


logger = logging.getLogger(__name__) # logging per module
#logger.setLevel(logging.INFO)


@typeguard.typechecked
def set_file_logger( # hacky
        path_log: Union[Path, str],
        level: Union[str, int], # 'DEBUG' or logging.DEBUG
        ):
    formatter = logging.Formatter(fmt='%(levelname)s - %(name)s - %(message)s')
    handler = logging.FileHandler(path_log)
    handler.setFormatter(formatter)
    names = [
            'psiflow.checks',
            'psiflow.data',
            'psiflow.ensemble',
            'psiflow.execution',
            'psiflow.experiment',
            'psiflow.learning',
            'psiflow.utils',
            'psiflow.models.base',
            'psiflow.models._mace',
            'psiflow.models._nequip',
            'psiflow.reference._cp2k',
            ]
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)


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
def get_psiflow_config_from_file(
        path_config: Union[Path, str],
        path_internal: Union[Path, str],
        ) -> tuple[Config, dict]:
    path_config = Path(path_config)
    assert path_config.is_file()
    # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    spec = importlib.util.spec_from_file_location('module.name', path_config)
    psiflow_config_module = importlib.util.module_from_spec(spec)
    sys.modules['module.name'] = psiflow_config_module
    spec.loader.exec_module(psiflow_config_module)
    return psiflow_config_module.get_config(path_internal)


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
    import shutil
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
    #shutil.rmtree(path_wandb)
log_data_to_wandb = python_app(
        _log_data_to_wandb,
        executors=['default'],
        cache=True,
        )


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


class SlurmProvider(parsl.providers.slurm.slurm.SlurmProvider):

    def submit(self, command, tasks_per_node, job_name="parsl.slurm"):
        """Submit the command as a slurm job.

        This function differs in its parent in the self.execute_wait()
        call, in which the slurm partition is explicitly passed as a command
        line argument as this is necessary for some SLURM-configered systems
        (notably, Belgium's HPC infrastructure).
        In addition, the way in which the job_id is extracted from the returned
        log after submission is slightly modified, again to account for
        the specific cluster configuration of HPCs in Belgium.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        tasks_per_node : int
            Command invocations to be launched per node
        job_name : str
            Name for the job
        Returns
        -------
        None or str
            If at capacity, returns None; otherwise, a string identifier for the job
        """

        scheduler_options = self.scheduler_options
        worker_init = self.worker_init
        if self.mem_per_node is not None:
            scheduler_options += '#SBATCH --mem={}g\n'.format(self.mem_per_node)
            worker_init += 'export PARSL_MEMORY_GB={}\n'.format(self.mem_per_node)
        if self.cores_per_node is not None:
            cpus_per_task = math.floor(self.cores_per_node / tasks_per_node)
            scheduler_options += '#SBATCH --cpus-per-task={}'.format(cpus_per_task)
            worker_init += 'export PARSL_CORES={}\n'.format(cpus_per_task)

        job_name = "{0}.{1}".format(job_name, time.time())

        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)


        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = tasks_per_node
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["scheduler_options"] = scheduler_options
        job_config["worker_init"] = worker_init
        job_config["user_script"] = command

        # Wrap the command
        job_config["user_script"] = self.launcher(command,
                                                  tasks_per_node,
                                                  self.nodes_per_block)

        self._write_submit_script(template_string, script_path, job_name, job_config)

        if self.move_files:
            channel_script_path = self.channel.push_file(script_path, self.channel.script_dir)
        else:
            channel_script_path = script_path

        retcode, stdout, stderr = self.execute_wait("sbatch --partition={1} {0}".format(channel_script_path, self.partition))

        job_id = None
        if retcode == 0:
            for line in stdout.split('\n'):
                if line.startswith("Submitted batch job"):
                    #job_id = line.split("Submitted batch job")[1].strip()
                    job_id = line.split("Submitted batch job")[1].strip().split()[0]
                    self.resources[job_id] = {'job_id': job_id, 'status': JobStatus(JobState.PENDING)}
        else:
            logger.error("Submit command failed")
            logger.error("Retcode:%s STDOUT:%s STDERR:%s", retcode, stdout.strip(), stderr.strip())
        return job_id
