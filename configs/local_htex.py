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
            HighThroughputExecutor(
                address='localhost',
                label='training',
                working_dir=str(path_internal / 'training_executor'),
                provider=provider,
                max_workers=1,
                ),
            HighThroughputExecutor(
                address='localhost',
                label='default',
                working_dir=str(path_internal / 'default_executor'),
                provider=provider,
                cores_per_worker=1,
                ),
            HighThroughputExecutor(
                address='localhost',
                label='model',
                working_dir=str(path_internal / 'model_executor'),
                provider=provider,
                ),
            HighThroughputExecutor(
                address='localhost',
                label='reference',
                working_dir=str(path_internal / 'reference_executor'),
                provider=provider,
                max_workers=1,
                cores_per_worker=4,
                ),
            ]
    return Config(executors, run_dir=str(path_internal), usage_tracking=True)
