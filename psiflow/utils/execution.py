from pathlib import Path

import typeguard
from parsl.executors import WorkQueueExecutor
from parsl.launchers.base import Launcher


@typeguard.typechecked
def container_launch_command(
    uri: str,
    engine: str = "apptainer",
    gpu: bool = False,
    addopts: str = " --no-eval -e --no-mount home -W /tmp --writable-tmpfs",
    entrypoint: str = "/opt/entry.sh",
) -> str:
    assert engine in ["apptainer", "singularity"]
    assert len(uri) > 0

    launch_command = ""
    launch_command += engine
    launch_command += " exec "
    launch_command += addopts
    launch_command += " --bind {}".format(
        Path.cwd().resolve()
    )  # access to data / internal dir
    if gpu:
        if "rocm" in uri:
            launch_command += " --rocm"
        else:  # default
            launch_command += " --nv"
    launch_command += " {} {} ".format(uri, entrypoint)
    return launch_command


class SlurmLauncher(Launcher):
    def __init__(self, debug: bool = True, overrides: str = ""):
        super().__init__(debug=debug)
        self.overrides = overrides

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        x = """set -e

NODELIST=$(scontrol show hostnames)
NODE_ARRAY=($NODELIST)
NODE_COUNT=${{#NODE_ARRAY[@]}}
EXPECTED_NODE_COUNT={nodes_per_block}

# Check if the length of NODELIST matches the expected number of nodes
if [ $NODE_COUNT -ne $EXPECTED_NODE_COUNT ]; then
  echo "Error: Expected $EXPECTED_NODE_COUNT nodes, but got $NODE_COUNT nodes."
  exit 1
fi

for NODE in $NODELIST; do
  srun --nodes=1 --ntasks=1 --exact -l {overrides} --nodelist=$NODE {command} &
  if [ $? -ne 0 ]; then
    echo "Command failed on node $NODE"
  fi
done

wait
""".format(
            nodes_per_block=nodes_per_block,
            command=command,
            overrides=self.overrides,
        )
        return x


class MyWorkQueueExecutor(WorkQueueExecutor):
    def _get_launch_command(self, block_id):
        return self.worker_command
