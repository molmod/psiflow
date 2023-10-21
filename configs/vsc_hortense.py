from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

from psiflow.execution import (
    Default,
    ModelTraining,
    ModelEvaluation,
    ReferenceEvaluation,
    generate_parsl_config,
)
from psiflow.parsl_utils import ContainerizedLauncher


launcher_cpu = ContainerizedLauncher(
    "oras://ghcr.io/molmod/psiflow:2.0.0-cuda",
    apptainer_or_singularity="apptainer",
    enable_gpu=False,
)
launcher_gpu = ContainerizedLauncher(
    "oras://ghcr.io/molmod/psiflow:2.0.0-cuda",
    apptainer_or_singularity="apptainer",
    enable_gpu=True,
)
# launcher_cpu = SimpleLauncher()
# launcher_gpu = SimpleLauncher()

default = Default(
    cores_per_worker=4,
    parsl_provider=SlurmProvider(
        partition="cpu_rome",
        account="2022_050",
        nodes_per_block=1,  # each block fits on (less than) one node
        cores_per_node=8,  # number of cores per slurm job
        init_blocks=1,  # initialize a block at the start of the workflow
        min_blocks=1,  # always keep at least one block open
        max_blocks=1,  # do not use more than one block
        walltime="72:00:00",  # walltime per block
        exclusive=False,
        scheduler_options="#SBATCH --clusters=dodrio\n",
        launcher=launcher_cpu,
    ),
)
model_evaluation = ModelEvaluation(
    cores_per_worker=12,
    max_walltime=None,  # kill gracefully before end of slurm job
    simulation_engine="openmm",
    gpu=True,
    parsl_provider=SlurmProvider(
        partition="gpu_rome_a100",
        account="2023_006",
        nodes_per_block=1,
        cores_per_node=12,
        init_blocks=0,
        max_blocks=32,
        walltime="12:00:00",
        exclusive=False,
        scheduler_options="#SBATCH --gpus=1\n#SBATCH --clusters=dodrio",
        launcher=launcher_gpu,
    ),
)
model_training = ModelTraining(
    cores_per_worker=12,
    gpu=True,
    max_walltime=None,  # kill gracefully before end of slurm job
    parsl_provider=SlurmProvider(
        partition="gpu_rome_a100",
        account="2023_006",
        nodes_per_block=1,
        cores_per_node=12,
        init_blocks=0,
        max_blocks=4,
        walltime="12:00:00",
        exclusive=False,
        scheduler_options="#SBATCH --gpus=1\n#SBATCH --clusters=dodrio",
        launcher=launcher_gpu,
    ),
)
reference_evaluation = ReferenceEvaluation(
    cores_per_worker=64,
    mpi_command=lambda x: f"mpirun -np {x} -bind-to core -rmk user -launcher fork",
    max_walltime=20,  # singlepoints should finish in less than 20 mins
    parsl_provider=SlurmProvider(
        partition="cpu_milan",
        account="2022_050",
        nodes_per_block=1,
        cores_per_node=64,  # 1 reference evaluation per SLURM job
        init_blocks=0,
        max_blocks=12,
        walltime="12:00:00",
        exclusive=True,
        scheduler_options="#SBATCH --clusters=dodrio\n",
        launcher=launcher_cpu,
    ),
)


def get_config(path_internal):
    definitions = [
        default,
        model_evaluation,
        model_training,
        reference_evaluation,
    ]
    config = generate_parsl_config(
        path_internal,
        definitions,
        use_work_queue=False,
        parsl_max_idletime=10,
        parsl_retries=1,
        parsl_strategy="htex_auto_scale",
    )
    return config, definitions
