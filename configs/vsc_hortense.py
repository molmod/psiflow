from psiflow.external import SlurmProviderVSC # fixed SlurmProvider

from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference, HybridCP2KReference, \
        MP2CP2KReference, DoubleHybridCP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config, ContainerizedLauncher


# psiflow definitions
model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=4,
        dtype='float32',
        walltime=60,
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=12,  # number of cores per GPU on gpu_rome_a100 partition
        walltime=60, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=64,              # number of cores per singlepoint
        omp_num_threads=1,      # only use MPI for parallelization
        mpi_command=lambda x: f'mpirun -np {x} -bind-to rr',
        cp2k_exec='cp2k.psmp',
        walltime=20,            # minimum walltime per singlepoint
        )
definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        AllegroModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        MP2CP2KReference: [reference_evaluate],
        HybridCP2KReference: [reference_evaluate],
        DoubleHybridCP2KReference: [reference_evaluate],
        }


providers = {}

launcher_cpu = ContainerizedLauncher(
        'oras://ghcr.io/molmod/psiflow:1.0.0-cuda11.3',
        apptainer_or_singularity='apptainer',
        enable_gpu=False,
        )
launcher_gpu = ContainerizedLauncher(
        'oras://ghcr.io/molmod/psiflow:1.0.0-cuda11.3',
        apptainer_or_singularity='apptainer',
        enable_gpu=True,
        )

cluster = 'dodrio' # all partitions reside on a single cluster

# define provider for default executor (HTEX)
provider = SlurmProviderVSC(       # one block == one slurm job to submit
        cluster=cluster,
        partition='cpu_rome',
        account='2022_069',
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=1,       # number of cores per slurm job, 1 is OK
        init_blocks=1,          # initialize a block at the start of the workflow
        min_blocks=1,           # always keep at least one block open
        max_blocks=1,           # do not use more than one block
        walltime='24:00:00',    # walltime per block
        exclusive=False,
        launcher=launcher_cpu,
        )
providers['default'] = provider


# define provider for executing model evaluations (e.g. MD)
provider = SlurmProviderVSC(
        cluster=cluster,
        partition='cpu_rome',
        account='2022_069',
        nodes_per_block=1,
        cores_per_node=4,
        init_blocks=0,
        min_blocks=0,
        max_blocks=512,
        parallelism=1,
        walltime='02:00:00',
        exclusive=False,
        launcher=launcher_cpu,
        )
providers['model'] = provider


# define provider for executing model training
provider = SlurmProviderVSC(
        cluster=cluster,
        partition='gpu_rome_a100',
        account='2022_069',
        nodes_per_block=1,
        cores_per_node=12,
        init_blocks=0,
        min_blocks=0,
        max_blocks=4,
        parallelism=1.0,
        walltime='01:05:00',
        worker_init='ml CUDA/11.7\n',
        exclusive=False,
        scheduler_options='#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu=12\n', # request gpu
        launcher=launcher_gpu,
        )
providers['training'] = provider


provider = SlurmProviderVSC(
        cluster=cluster,
        partition='cpu_rome',
        account='2022_069',
        nodes_per_block=1,
        cores_per_node=reference_evaluate.ncores, # 1 worker per block; leave this
        init_blocks=0,
        min_blocks=0,
        max_blocks=10,
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
            wq_timeout=120,         # timeout for WQ workers before they shut down
            parsl_app_cache=False,  # parsl app caching; disabled for safety
            parsl_retries=1,        # HTEX may fail when block hits walltime
            parsl_max_idletime=30,  # idletime before parsl tries to scale-in resources
            )
    return config, definitions
