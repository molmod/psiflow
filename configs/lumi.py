from psiflow.external import SlurmProviderVSC # fixed SlurmProvider

from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config, ContainerizedLauncher


# psiflow definitions
model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cuda',      # run MD on GPU
        ncores=8,
        dtype='float32',
        walltime=30, # in minutes
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=8, # number of cores per GPU
        walltime=120, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=64,          # number of cores per singlepoint
        omp_num_threads=1,  # only use MPI for parallelization
        mpi_command=lambda x: f'mpirun -np {x} -bind-to rr',
        cp2k_exec='cp2k.psmp',
        walltime=15,         # maximum walltime per singlepoint
        )
definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        AllegroModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        }


providers = {}

launcher_cpu = ContainerizedLauncher(
        'oras://ghcr.io/molmod/psiflow:1.0.0-rocm5.2',
        apptainer_or_singularity='singularity',
        enable_gpu=False,
        )
launcher_gpu = ContainerizedLauncher(
        'oras://ghcr.io/molmod/psiflow:1.0.0-rocm5.2',
        apptainer_or_singularity='singularity',
        enable_gpu=True,
        )

# define provider for default executor (HTEX)
# each of the workers in this executor is single-core;
# they do basic processing stuff (reading/writing data/models, ... )
worker_init = 'export SINGULARITYENV_WANDB_CACHE_DIR="$(pwd)"\n'
provider = SlurmProviderVSC(       # one block == one slurm job to submit
        cluster='lumi',
        partition='small',
        account='project_465000315',
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=16,      # number of cores per slurm job, 1 is OK
        init_blocks=1,          # initialize a block at the start of the workflow
        min_blocks=1,           # always keep at least one block open
        max_blocks=1,           # do not use more than one block
        walltime='24:00:00',    # walltime per block
        exclusive=False,
        worker_init=worker_init,
        launcher=launcher_cpu,
        )
providers['default'] = provider


# define provider for executing model training and inference (both GPU)
worker_init = 'ml LUMI/22.08\n'
worker_init += 'ml rocm/5.2.3\n'
worker_init += 'export SINGULARITY_BIND="/opt/rocm"\n' # --rocm flag doesn't bind everything
worker_init += 'export SINGULARITYENV_ROCM_PATH="/opt/rocm"\n'
worker_init += 'export SINGULARITYENV_ROCBLAS_TENSILE_LIBPATH="/opt/rocm/lib/rocblas/library/"\n'
worker_init += 'export SINGULARITYENV_XTPE_LINK_TYPE="dynamic\n"'
provider = SlurmProviderVSC(
        cluster='lumi',
        partition='eap',
        account='project_465000315',
        nodes_per_block=1,
        cores_per_node=32, # 4 GPUs per block; 4 workers per job
        init_blocks=0,
        min_blocks=0,
        max_blocks=5,
        parallelism=1.0,
        walltime='02:05:00',
        worker_init=worker_init,
        exclusive=False,
        scheduler_options='#SBATCH --gpus=4\n#SBATCH --cpus-per-gpu=8\n', # request gpu
        launcher=launcher_gpu,
        )
providers['training'] = provider
provider = SlurmProviderVSC(
        cluster='lumi',
        partition='eap',
        account='project_465000315',
        nodes_per_block=1,
        cores_per_node=32, # 4 GPUs per SLURM job; 4 workers per job
        init_blocks=0,
        min_blocks=0,
        max_blocks=4,
        parallelism=1.0,
        walltime='01:05:00',
        worker_init=worker_init,
        exclusive=False,
        scheduler_options='#SBATCH --gpus=4\n#SBATCH --cpus-per-gpu=8\n', # request gpu
        launcher=launcher_gpu,
        )
providers['model'] = provider


provider = SlurmProviderVSC(
        cluster='lumi',
        partition='small',
        account='project_465000315',
        nodes_per_block=1,
        cores_per_node=reference_evaluate.ncores, # 1 worker per block; leave this!
        init_blocks=0,
        min_blocks=0,
        max_blocks=20,
        parallelism=1,
        walltime='00:59:59',
        exclusive=False,
        launcher=launcher_cpu,
        )
providers['reference'] = provider


def get_config(path_parsl_internal):
    config = generate_parsl_config(
            path_parsl_internal,
            definitions,
            providers,
            use_work_queue=True,
            wq_timeout=30,          # timeout for WQ workers before they shut down
            parsl_app_cache=False,  # parsl app caching; disabled for safety
            parsl_retries=1,        # single retry
            parsl_max_idletime=20,  # idletime before parsl tries to scale-in resources
            )
    return config, definitions
