from psiflow.utils import SlurmProvider, LocalChannel # fixed SlurmProvider

from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config


# psiflow definitions
model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=4,
        dtype='float32',
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=12, # number of cores per GPU on gpu_rome_a100 partition
        walltime=30, # in minutes; includes 100s slack
        )
mpi_stevin = 'mpirun --mca pml ucx --mca btl ^uct --mca orte_keep_fqdn_hostnames 1 -np ${SLURM_NTASKS} --map-by ppr:${SLURM_CPUS_ON_NODE}:node:PE=1:SPAN:NOOVERSUBSCRIBE'
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=32,          # number of cores per singlepoint
        omp_num_threads=1,  # only use MPI for parallelization
        mpi_command=lambda x: mpi_stevin,
        cp2k_exec='cp2k.psmp',  # on some platforms, this is cp2k.popt
        walltime=30,            # minimum walltime per singlepoint
        )
definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        AllegroModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        }


providers = {}


# define provider for default executor (HTEX)
# each of the workers in this executor is single-core;
# they do basic processing stuff (reading/writing data/models, ... )
cluster = 'doduo'
worker_init = ''
worker_init += 'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
provider = SlurmProvider(
        channel=LocalChannel(prepend_cmd='module swap cluster/{}'.format(cluster)),
        partition=cluster,      # redundant specification of partition is necessary!
        nodes_per_block=1,      # each block fits on (less than) one node
        cores_per_node=1,       # number of cores per slurm job
        init_blocks=1,          # initialize a block at the start of the workflow
        min_blocks=1,           # always keep at least one block open
        max_blocks=1,           # do not use more than one block
        walltime='02:00:00',    # walltime per block
        worker_init=worker_init,
        scheduler_options='#SBATCH --export=NONE\n',
        exclusive=False,
        )
providers['default'] = provider


# define provider for executing model evaluations (e.g. MD)
cluster = 'doduo'
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml unload SciPy-bundle/2021.05-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
worker_init += 'export OMP_NUM_THREADS={}\n'.format(model_evaluate.ncores)
provider = SlurmProvider(
        channel=LocalChannel(prepend_cmd='module swap cluster/{}'.format(cluster)),
        partition=cluster,
        nodes_per_block=1,
        cores_per_node=4,
        init_blocks=0,
        min_blocks=0,
        max_blocks=512,
        parallelism=1,
        walltime='02:00:00',
        worker_init=worker_init,
        scheduler_options='#SBATCH --export=NONE\n',
        exclusive=False,
        )
providers['model'] = provider


# define provider for executing model training
cluster = 'accelgor'
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CUDA-11.3.1\n'
worker_init += 'unset SLURM_CPUS_PER_TASK\n'
worker_init += 'export SLURM_NTASKS_PER_NODE={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_NTASKS={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_NPROCS={}\n'.format(model_training.ncores)
worker_init += 'export OMP_NUM_THREADS={}\n'.format(model_training.ncores)
provider = SlurmProvider(
        channel=LocalChannel(prepend_cmd='module swap cluster/{}'.format(cluster)),
        partition=cluster,
        nodes_per_block=1,
        cores_per_node=12,
        init_blocks=0,
        min_blocks=0,
        max_blocks=4,
        parallelism=1.0,
        walltime='01:05:00',
        worker_init=worker_init,
        scheduler_options='#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu=12\n#SBATCH --export=None', # request gpu
        exclusive=False,
        )
providers['training'] = provider


# to get MPI to recognize the available slots correctly, it's necessary
# to override the slurm variables as set by the jobscript, as these are
# based on the number of parsl tasks, NOT on the number of MPI tasks for
# cp2k. Essentially, this means we have to reproduce the environment as
# if we launched a job using 'qsub -l nodes=1:ppn=cores_per_singlepoint'
# The custom mpirun wrapper on the VSC Stevin infrastructure is built
# using Python 3.6, which does not go well with psiflow and/or CP2K
cluster = 'doduo'
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml CP2K/8.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
worker_init += 'unset SLURM_CPUS_PER_TASK\n'
worker_init += 'export SLURM_NTASKS_PER_NODE={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_NTASKS={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_NPROCS={}\n'.format(reference_evaluate.ncores)
provider = SlurmProvider(
        channel=LocalChannel(prepend_cmd='module swap cluster/{}'.format(cluster)),
        partition=cluster,
        nodes_per_block=1,
        cores_per_node=reference_evaluate.ncores, # 1 worker per block; leave this
        init_blocks=0,
        min_blocks=0,
        max_blocks=10,
        parallelism=1,
        walltime='01:00:00',
        worker_init=worker_init,
        scheduler_options='#SBATCH --export=NONE\n',
        exclusive=False,
        )
providers['reference'] = provider


def get_config(path_parsl_internal):
    config = generate_parsl_config(
            path_parsl_internal,
            definitions,
            providers,
            use_work_queue=True,
            wq_timeout=120,         # timeout for WQ workers before they shut down
            wq_port=9223,           # start of port range used by WQ executors
            parsl_app_cache=False,  # parsl app caching; disabled for safety
            parsl_retries=1,        # HTEX may fail when block hits walltime
            parsl_max_idletime=30,  # idletime before parsl tries to scale-in resources
            )
    return config, definitions
