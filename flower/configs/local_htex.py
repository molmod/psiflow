from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider
from parsl.config import Config


def get_config(path_internal):
    provider = LocalProvider(
        min_blocks=1,
        max_blocks=1,
        nodes_per_block=1,
        parallelism=0.5,
        launcher=SingleNodeLauncher(),
        )
    executors = [
            HighThroughputExecutor(address='localhost', label='gpu', working_dir=str(path_internal), provider=provider, max_workers=1),
            HighThroughputExecutor(address='localhost', label='default', working_dir=str(path_internal), provider=provider, cores_per_worker=1),
            HighThroughputExecutor(address='localhost', label='cpu_small', working_dir=str(path_internal), provider=provider),
            HighThroughputExecutor(address='localhost', label='cpu_large', working_dir=str(path_internal), provider=provider, max_workers=1, cores_per_worker=4),
            ]
    return Config(executors, run_dir=str(path_internal))
