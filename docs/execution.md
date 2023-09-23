Psiflow makes it extremely easy to build
complex computational graphs that consist of
QM evaluations, model training, and a variety of phase space sampling algorithms, among others.
If your Python environment contains the required dependencies, you can execute any of the learning examples just like that:
```console
$ python zeolite_reaction.py
```
Model training, molecular dynamics, and reference evaluations will all get executed in separate processes; the main
Python script only used to resolve the computational directed acyclic graph (DAG) of tasks.
However, local execution is rather limiting because the evaluation of an average learning workflow requires
large amounts of several different computational resources.
For example, QM calculations typically require nodes with a large core count (64 or even 128)
and with sufficient memory, whereas model training and evaluation require one or more powerful
GPUs.
Because a single computer can never provide the computing power that is required to
execute such workflows, psiflow is intrinsically built to support **distributed
and asynchronous execution** across a large variety of resources (including most HPC and cloud infrastructure).
This means that while the entire online learning workflow is defined in a single Python script,
its execution is automatically offloaded to tens, hundreds, or even thousands of nodes.
Configuration of the execution resources is done using a single Python script; `config.py`.
It specifies the following parameters (among others)

- the number of cores to use for each CP2K singlepoint evaluation, as well as the specific OpenMP/MPI parallellization settings;
- the project ID from Google Compute Engine, if applicable;
- the computational resources to request in a single SLURM job, if applicable.
This includes walltime, core count, and memory, as well as e.g. the maximum number of molecular dynamics
simulations that can be executed in parallel in a single job;
- the execution of specific `module load` or `source env/bin/activate` commands to ensure all the necessary environment variables are set.

The execution parameters in the configuration script are strictly and deliberately kept separate
from the main Python script that defines the workflow, in line with Parsl's philosophy *write once, run anywhere*.
To execute the zeolite reaction example not on your local computer, but on remote compute resources
(e.g. the Frontier exascale system at OLCF), you simply have to __pass the relevant psiflow configuration
file as an argument__:

```console
  $ python my_workflow.py --psiflow-config frontier.py      # executes exact same workflow on Frontier
```

The following sections will explain in more detail how remote execution is configured.


!!! note "Parsl execution"
    Before you continue, we recommend going through the 
    [Parsl documentation on execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html)
    first in order to get acquainted with the `executor`, `provider`, and `launcher`
    concepts.

## Execution definitions
The definition of all execution-side parameters happens in the psiflow configuration file.
Its contents are divided in so-called execution definitions, each of which specifies how and where
a certain type of calculation will be executed. Each definition accepts at least the following arguments:

- `gpu: bool = False`: whether or not this calculation proceeds on the GPU
- `cores_per_worker: int = 1`: how many cores each individual calculation requires
- `max_walltime: float = None`: specifies a maximum duration of each calculation before it gets gracefully killed.
- `parsl_provider: parsl.providers.ExecutionProvider`: a Parsl provider which psiflow can use to get compute time.
For a `ClusterProvider`, this involves submitting a job to the queueing system; for a `GoogleCloudProvider`,
this involves provisioning and connecting to a node in the cloud; for a `LocalProvider`, this just means "use the resources
available on the current system". See [this section](https://parsl.readthedocs.io/en/stable/userguide/execution.html#execution-providers)
in the Parsl documentation for more details.

Psiflow introduces four different execution definitions:

1. __model evaluation__: this determines how and where the trainable models (i.e. `BaseModel` instances) are
executed. In addition to the common arguments above, it allows the user to specify a `simulation_engine`, with possible
values being `yaff` (slow, legacy) and `openmm` (faster, new in v2.0.0). Older versions additionally allowed users to
perform certain calculations in `float64` (which might be relevant when doing energy minimizations) but this has been
removed in v2.0.0; all calculations are now performed in `float32`.
2. __model training__: this definition determines where models are trained. It is _forced_ to have `gpu=True` because model training
is always performed on GPU (and in `float32`) anyway.
3. __reference evaluation__: this determines how and where QM evaluations are performed. Because most (if not all) QM engines
rely on MPI (and sometimes OpenMP) to parallelize over multiple cores, it is possible to specify your own MPI command here.
It should be a function with a single argument, which returns (as string) the MPI command to use when executing a QM evaluation.
Its default value is the following `lambda` expression (for MPICH as included in the container):
```py
mpi_command = lambda x: f'mpirun -np {x} -bind-to core -rmk user -launcher fork'
```
4. __default execution__: this determines where the lightweight/administrative operations get executed (dataset copying, basic numpy operations, ...).
Typically, this requires only a few cores and a few GBs of memory.

## Configuration
Let's illustrate how the execution definitions can be used to construct a psiflow configuration file.
Essentially, such files consist of a single `get_config()` method which should return a Parsl `Config` object
as well as a list of psiflow `ExecutionDefinition` instances as discussed above.
### Local
For simplicity, let us assume for now that all of the required compute resources are available locally,
which means we can use parsl's `LocalProvider` in the various definitions:
```py title="local_htex.py"
from parsl.providers import LocalProvider

from psiflow.execution import Default, ModelTraining, ModelEvaluation, \
        ReferenceEvaluation, generate_parsl_config


default = Default()                         # subclass of ExecutionDefinition
model_evaluation = ModelEvaluation(         # subclass of ExecutionDefinition
        parsl_provider=LocalProvider(),
        cores_per_worker=2,
        max_walltime=30,                    # in minutes!
        simulation_engine='openmm',         # or yaff; openmm is faster
        gpu=True,                       
        )
model_training = ModelTraining(             # subclass of ExecutionDefinition
        parsl_provider=LocalProvider(),     # or SlurmProvider / GoogleCloudProvider / ...
        gpu=True,
        max_walltime=10,                
        )
reference_evaluation = ReferenceEvaluation( # subclass of ExecutionDefinition
        parsl_provider=LocalProvider(),
        cores_per_worker=6,
        max_walltime=60,                    # kill after this amount of time, in minutes
        mpi_command=lambda x: f'mpirun -np {x} -bind-to core -rmk user -launcher fork',
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
            use_work_queue=False,           # can improve the scheduling of jobs but also slows down
            parsl_max_idletime=20,
            parsl_retries=1,                # retry at least once in case the error happened randomly
            )
    return config, definitions

```
These definitions imply that:

- molecular dynamics will be performed using the OpenMM engine, one two cores and one GPU,
and will be gracefully killed after 30 minutes;
- model training is allowed to run for a maximum of 10 minutes;
- reference QM evaluations will be performed using 6 cores per singlepoint, and with a specific MPI command.
If a singlepoint takes longer than 60 minutes, it is killed. No energy labels will be stored in the state,
and the corresponding FlowAtoms instance will have `reference_status = False`.

The `get_config()` method takes a single argument (a cache directory shared by all compute resources) and builds a
Parsl `Config` based on the provided definitions and a few additional parameters, such the maximum number of retries
that Parsl may attempt for a specific task
(which can be useful e.g. when the cluster you're using contains a few faulty nodes).


### Remote (HPC/Cloud)

As mentioned before, psiflow is designed to support remote execution on vast amounts of
compute resources.
This is particularly convenient in a containerized fashion, since this alleviates the
need to install all of its dependencies on each of the compute resources.
As an example of this, consider the following configuration:

```py title="vsc_hortense.py"
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

from psiflow.execution import Default, ModelTraining, ModelEvaluation, \
        ReferenceEvaluation, generate_parsl_config
from psiflow.parsl_utils import ContainerizedLauncher


# The ContainerizedLauncher is a subclass of Parsl Launchers, which simply
# wraps all commands to be executed inside a container.
# The ORAS containers are downloaded from Github -- though it's best to cache
# them beforehand (e.g. by executing 'apptainer exec <uri> pwd').
launcher_cpu = ContainerizedLauncher(
        'docker://ghcr.io/molmod/psiflow:2.0.0-cuda11.8',
        apptainer_or_singularity='apptainer',
        )
launcher_gpu = ContainerizedLauncher(
        'docker://ghcr.io/molmod/psiflow:2.0.0-cuda11.8',
        apptainer_or_singularity='apptainer',
        enable_gpu=True,            # binds GPU in container
        )

default = Default(
        cores_per_worker=4,         # 8 / 4 = 2 workers per slurm job
        parsl_provider=SlurmProvider(
            partition='cpu_rome',
            account='2022_050',
            nodes_per_block=1,      # each block fits on (less than) one node
            cores_per_node=8,       # number of cores per slurm job
            init_blocks=1,          # initialize a block at the start of the workflow
            min_blocks=1,           # always keep at least one block open
            max_blocks=1,           # do not use more than one block
            walltime='72:00:00',    # walltime per block
            exclusive=False,
            scheduler_options='#SBATCH --clusters=dodrio\n', # specify the cluster
            launcher=launcher_cpu,  # no GPU needed
            )
        )
model_evaluation = ModelEvaluation(
        cores_per_worker=12,        # ncores per GPU
        max_walltime=None,          # kill gracefully before end of slurm job
        simulation_engine='openmm',
        gpu=True,
        parsl_provider=SlurmProvider(
            partition='gpu_rome_a100',
            account='2023_006',
            nodes_per_block=1,
            cores_per_node=12,
            init_blocks=0,
            max_blocks=32,
            walltime='12:00:00',
            exclusive=False,
            scheduler_options='#SBATCH --gpus=1\n#SBATCH --clusters=dodrio', # ask for a GPU!
            launcher=launcher_gpu,  # binds GPU in container!
            )
        )
model_training = ModelTraining(
        cores_per_worker=12,
        gpu=True,
        max_walltime=None,          # kill gracefully before end of slurm job
        parsl_provider=SlurmProvider(
            partition='gpu_rome_a100',
            account='2023_006',
            nodes_per_block=1,
            cores_per_node=12,
            init_blocks=0,
            max_blocks=4,
            walltime='12:00:00',
            exclusive=False,
            scheduler_options='#SBATCH --gpus=1\n#SBATCH --clusters=dodrio',
            launcher=launcher_gpu,
            )
        )
reference_evaluation = ReferenceEvaluation(
        cores_per_worker=64,
        max_walltime=20,            # singlepoints should finish in less than 20 mins
        parsl_provider=SlurmProvider(
            partition='cpu_milan',
            account='2022_050',
            nodes_per_block=1,
            cores_per_node=64,      # 1 reference evaluation per SLURM job
            init_blocks=0,
            max_blocks=12,
            walltime='12:00:00',
            exclusive=True,
            scheduler_options='#SBATCH --clusters=dodrio\n',
            launcher=launcher_cpu,
            )
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
            )
    return config, definitions
```
Check out the [configs](https://github.com/molmod/psiflow/tree/main/configs) directory for more example configurations.

For people who chose to install psiflow and its dependencies [manually](installation.md#manual),
there's no need to define custom Parsl Launchers.
Instead, they need to care that all manually installed packages can be found by each of the workers;
this is possible using the `worker_init` argument of Parsl providers, which can be used to 
activate specific Python environments or execute `module load` commands.
