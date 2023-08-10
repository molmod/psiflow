import typeguard
import os
import math
import time
import logging
from pathlib import Path
from importlib import metadata # necessary on python 3.9

from typing import Optional

from parsl.executors import WorkQueueExecutor
from parsl.launchers.launchers import Launcher


logger = logging.getLogger(__name__)


class MyWorkQueueExecutor(WorkQueueExecutor):
    
    def _get_launch_command(self, block_id):
        return self.worker_command


VERSION    = metadata.version('psiflow')
ADDOPTS    = ' --no-eval -e --no-mount home -W /tmp --writable-tmpfs'
ENTRYPOINT = '/usr/local/bin/entry.sh'


@typeguard.typechecked
class ContainerizedLauncher(Launcher):

    def __init__(
        self,
        uri: str,
        apptainer_or_singularity: str = 'apptainer',
        addopts: str = ADDOPTS,
        entrypoint: str = ENTRYPOINT,
        enable_gpu: Optional[bool] = False,
    ) -> None:
        super().__init__(debug=True)
        self.uri = uri # required by Parsl parent class to assign attributes
        self.apptainer_or_singularity = apptainer_or_singularity
        self.addopts = addopts
        self.entrypoint = entrypoint
        self.enable_gpu = enable_gpu

        self.launch_command = ''
        self.launch_command += apptainer_or_singularity
        self.launch_command += ' exec '
        self.launch_command += addopts
        self.launch_command += ' --bind {}'.format(Path.cwd().resolve()) # access to data / internal dir
        env  = {}
        keys = ['WANDB_API_KEY']
        for key in keys:
            if key in os.environ.keys():
                env[key] = os.environ[key]
        if 'WANDB_API_KEY' not in env.keys():
            logger.critical('wandb API key not set; please go to wandb.ai/authorize and '
                'set that key in the current environment: export WANDB_API_KEY=<key-from-wandb.ai/authorize>')
        env['PARSL_CORES'] = '${PARSL_CORES}'
        if len(env) > 0:
            self.launch_command += ' --env '
            self.launch_command += ','.join([f'{k}={v}' for k, v in env.items()])
        if enable_gpu:
            if 'cuda' in self.uri:
                self.launch_command += ' --nv'
            else:
                self.launch_command += ' --rocm'
        self.launch_command += ' ' + uri + ' ' + entrypoint + ' '

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        return self.launch_command + "{}".format(command)
