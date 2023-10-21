from parsl.providers import LocalProvider
from parsl.executors import ThreadPoolExecutor
from parsl.config import Config

from psiflow.execution import (
    Default,
    ModelTraining,
    ModelEvaluation,
    ReferenceEvaluation,
)


default = Default(
    parsl_provider=LocalProvider(),  # unused
)
model_evaluation = ModelEvaluation(
    parsl_provider=LocalProvider(),
    cores_per_worker=2,
    max_walltime=1,
    simulation_engine="openmm",
    gpu=False,
)
model_training = ModelTraining(
    parsl_provider=LocalProvider(),
    gpu=True,
    max_walltime=1,
)
reference_evaluation = ReferenceEvaluation(
    parsl_provider=LocalProvider(),
    cores_per_worker=2,
    max_walltime=1.5,
    mpi_command=lambda x: f"mpirun -np {x}",
)
definitions = [
    default,
    model_evaluation,
    model_training,
    reference_evaluation,
]


def get_config(path_internal):
    executors = [
        ThreadPoolExecutor(
            label="Default",
            max_threads=default.cores_per_worker,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="ModelTraining",
            max_threads=model_training.cores_per_worker,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="ModelEvaluation",
            max_threads=model_evaluation.cores_per_worker,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="ReferenceEvaluation",
            max_threads=reference_evaluation.cores_per_worker,
            working_dir=str(path_internal),
        ),
    ]
    config = Config(
        executors, run_dir=str(path_internal), usage_tracking=True, app_cache=False
    )
    return config, definitions
