import logging
import os
from pathlib import Path
from typing import Optional

import typeguard
from parsl.launchers.launchers import Launcher

logger = logging.getLogger(__name__)


ADDOPTS = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs"
ENTRYPOINT = "/usr/local/bin/entry.sh"


@typeguard.typechecked
class ContainerizedLauncher(Launcher):
    def __init__(
        self,
        uri: str,
        engine: str = "apptainer",  # or singularity
        addopts: str = ADDOPTS,
        entrypoint: str = ENTRYPOINT,
        enable_gpu: Optional[bool] = False,
    ) -> None:
        super().__init__(debug=True)
        self.uri = uri  # required by Parsl parent class to assign attributes
        self.engine = engine
        self.addopts = addopts
        self.entrypoint = entrypoint
        self.enable_gpu = enable_gpu

        self.launch_command = ""
        self.launch_command += engine
        self.launch_command += " exec "
        self.launch_command += addopts
        self.launch_command += " --bind {}".format(
            Path.cwd().resolve()
        )  # access to data / internal dir
        env = {}
        keys = ["WANDB_API_KEY"]
        for key in keys:
            if key in os.environ.keys():
                env[key] = os.environ[key]
        if "WANDB_API_KEY" not in env.keys():
            logger.critical(
                "wandb API key not set; please go to wandb.ai/authorize and "
                "set that key in the current environment: export WANDB_API_KEY=<key-from-wandb.ai/authorize>"
            )
        env["PARSL_CORES"] = "${PARSL_CORES}"
        if len(env) > 0:
            self.launch_command += " --env "
            self.launch_command += ",".join([f"{k}={v}" for k, v in env.items()])
        if enable_gpu:
            if "cuda" in self.uri:
                self.launch_command += " --nv"
            else:
                self.launch_command += " --rocm"
        self.launch_command += " " + uri + " " + entrypoint + " "

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        return self.launch_command + "{}".format(command)


@typeguard.typechecked
class ContainerizedSrunLauncher(ContainerizedLauncher):
    def __init__(self, overrides: str = "", **kwargs):
        self.overrides = overrides
        super().__init__(**kwargs)

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        task_blocks = tasks_per_node * nodes_per_block
        debug_num = int(self.debug)

        x = """set -e
export CORES=$SLURM_CPUS_ON_NODE
export NODES=$SLURM_JOB_NUM_NODES

[[ "{debug}" == "1" ]] && echo "Found cores : $CORES"
[[ "{debug}" == "1" ]] && echo "Found nodes : $NODES"
WORKERCOUNT={task_blocks}

path_cmd=$(dirname $SLURM_JOB_STDOUT)

cat << SLURM_EOF > $path_cmd/cmd_$SLURM_JOB_NAME.sh
{command}
SLURM_EOF
chmod a+x $path_cmd/cmd_$SLURM_JOB_NAME.sh

srun --ntasks {task_blocks} -l {overrides} bash $path_cmd/cmd_$SLURM_JOB_NAME.sh

[[ "{debug}" == "1" ]] && echo "Done"
""".format(
            command=self.launch_command + "{}".format(command),
            task_blocks=task_blocks,
            overrides=self.overrides,
            debug=debug_num,
        )
        return x
