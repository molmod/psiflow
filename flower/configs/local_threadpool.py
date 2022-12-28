from parsl.executors import ThreadPoolExecutor
from parsl.config import Config


def get_config(path_internal):
    executors = [
            ThreadPoolExecutor(label='gpu', max_threads=1, working_dir=str(path_internal)),
            ThreadPoolExecutor(label='default', max_threads=1, working_dir=str(path_internal)),
            ThreadPoolExecutor(label='cpu_small', max_threads=4, working_dir=str(path_internal)),
            ThreadPoolExecutor(label='cpu_large', max_threads=4, working_dir=str(path_internal)),
            ]
    return Config(executors, run_dir=str(path_internal))
