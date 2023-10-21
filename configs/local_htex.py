from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher

from psiflow.execution import (
    Default,
    ModelTraining,
    ModelEvaluation,
    ReferenceEvaluation,
    generate_parsl_config,
)
from psiflow.parsl_utils import ContainerizedLauncher


containerize = False
if containerize:
    launcher = ContainerizedLauncher(
        uri="oras://ghcr.io/molmod/psiflow:2.0.0-cuda",
        enable_gpu=True,
    )
else:
    launcher = SimpleLauncher()


default = Default(
    parsl_provider=LocalProvider(launcher=launcher),  # unused
)
model_evaluation = ModelEvaluation(
    parsl_provider=LocalProvider(launcher=launcher),
    cores_per_worker=2,
    max_walltime=1,
    simulation_engine="yaff",
    gpu=True,
)
model_training = ModelTraining(
    parsl_provider=LocalProvider(launcher=launcher),
    gpu=True,
    max_walltime=1,
)
reference_evaluation = ReferenceEvaluation(
    parsl_provider=LocalProvider(launcher=launcher),
    cores_per_worker=2,
    max_walltime=1.5,
    mpi_command=lambda x: f"mpirun -np {x} -bind-to core -rmk user -launcher fork",
)
definitions = [
    default,
    model_evaluation,
    model_training,
    reference_evaluation,
]


def get_config(path_internal):
    config = generate_parsl_config(
        path_internal,
        definitions,
        use_work_queue=False,
        parsl_max_idletime=20,
    )
    return config, definitions
