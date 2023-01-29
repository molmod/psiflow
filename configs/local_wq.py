from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider
from parsl.config import Config


def get_config(path_internal):
    provider = LocalProvider(
        min_blocks=1,
        max_blocks=1,
        nodes_per_block=1,
        parallelism=1.0,
        launcher=SingleNodeLauncher(),
        )
    provider_reference = LocalProvider(
        min_blocks=1,
        max_blocks=1,
        nodes_per_block=1,
        parallelism=0.5,
        launcher=SingleNodeLauncher(),
        )
    executors = [
            HighThroughputExecutor(
                address='localhost',
                label='default',
                working_dir=str(path_internal / 'default_executor'),
                provider=provider,
                cores_per_worker=1,
                ),
            WorkQueueExecutor(
                label='training',
                working_dir=str(path_internal / 'training_executor'),
                provider=provider,
                shared_fs=True,
                autocategory=False,
                port=9123,
                max_retries=0,
                worker_options='--gpus=1 --cores=4', # 1min + eps
                ),
            WorkQueueExecutor(
                label='model',
                working_dir=str(path_internal / 'model_executor'),
                provider=provider,
                shared_fs=True,
                autocategory=False,
                port=9124,
                max_retries=0,
                worker_options='--gpus=0 --cores=1',
                ),
            # setting environment variables using the env argument did not work;
            # at least not for the omp num threads setting.
            WorkQueueExecutor(
                label='reference',
                working_dir=str(path_internal / 'reference_executor'),
                provider=provider,
                shared_fs=True,
                autocategory=False,
                port=9125,
                max_retries=0,
                init_command='export OMP_NUM_THREADS=1',
                worker_options='--gpus=0 --wall-time=60 --cores=4',
                ),
            ]
    return Config(executors, run_dir=str(path_internal), usage_tracking=True)
