Psiflow makes it extremely easy to build
complex computational graphs that consist of
QM evaluations, model training, and a variety of phase space sampling algorithms, among others.
If your Python environment contains the required dependencies, you can execute any of the learning examples just like that:
```console
python zeolite_reaction.py
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
python my_workflow.py frontier.yaml      # executes exact same workflow on Frontier
```

The following sections will explain in more detail how remote execution is configured.


!!! note "Parsl execution"
    
    Before you continue, we recommend going through the 
    [Parsl documentation on execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html)
    first in order to get acquainted with the `executor`, `provider`, and `launcher`
    concepts.

## Execution definitions
All execution-side parameters are defined in a configuration file using concise YAML syntax.
Its contents are divided in so-called execution definitions, each of which specifies how and where
a certain type of calculation will be executed. Each definition accepts at least the following arguments:

- `gpu: bool = False`: whether or not this calculation proceeds on the GPU
- `cores_per_worker: int = 1`: how many cores each individual calculation requires
- `max_walltime: float = None`: specifies a maximum duration of each calculation before it gets gracefully killed.
- `parsl_provider: parsl.providers.ExecutionProvider`: a Parsl provider which psiflow can use to get compute time.
For a `ClusterProvider` (e.g. `SlurmProvider`), this involves submitting a job to the queueing system; for a `GoogleCloudProvider`,
this involves provisioning and connecting to a node in the cloud; for a `LocalProvider`, this just means "use the resources
available on the current system". See [this section](https://parsl.readthedocs.io/en/stable/userguide/execution.html#execution-providers)
in the Parsl documentation for more details.

Psiflow introduces three different execution definitions:

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

For example: the following configuration ensures we're executing molecular dynamics sampling using OpenMM as engine and on 1 GPU / 4 cores,
whereas reference evaluation is performed on 4 cores. If a QM singlepoint takes longer than 10 minutes (because our system is too big or 
the SCF has trouble converging), the evaluation is gracefully killed.
```yaml
---
ModelEvaluation:
  cores_per_worker: 4
  simulation_engine: 'openmm'
  gpu: True
ModelTraining:
  gpu: true
ReferenceEvaluation:
  cores_per_worker: 4
  max_walltime: 10
...
```
whereas this one ensures we're using YAFF as backend, on a single core, without GPU.
In addition, it ensures that all nontrivial operations are executed using psiflow's container image 
(because you rightfully don't want to bother installing PyTorch/OpenMM/CP2K/PLUMED/... on your local workstation).
```yaml
---
container:
  engine: 'apptainer'
  uri: 'oras://ghcr.io/molmod/psiflow:3.0.0_python3.9_cuda'
ModelEvaluation:
  cores_per_worker: 1
  simulation_engine: 'yaff'
ModelTraining:
  gpu: true
ReferenceEvaluation:
  cores_per_worker: 4
  max_walltime: 10
...
```

## Remote execution
The above example is concise and elegant, but not very useful.
As mentioned before, psiflow is designed to support remote execution on vast amounts of
compute resources, not just the cores/GPU on our local workstation.
This is particularly convenient in a containerized fashion, since this alleviates the
need to install all of its dependencies on each of the compute resources.
As an example of this, consider the following configuration:

```yaml
---
container:
  engine: "apptainer"
  uri: "oras://ghcr.io/molmod/psiflow:3.0.0_python3.9_cuda"
ModelEvaluation:
  cores_per_worker: 1
  simulation_engine: 'openmm'
  SlurmProvider:
    partition: "cpu_rome"
    account: "2022_069"
    nodes_per_block: 1    # each block fits on (less than) one node
    cores_per_node: 8     # number of cores per slurm job
    init_blocks: 1        # initialize a block at the start of the workflow
    max_blocks: 1         # do not use more than one block
    walltime: "01:00:00"  # walltime per block
    exclusive: false      # rest of compute node free to use
    scheduler_options: "#SBATCH --clusters=dodrio\n"
ModelTraining:
  cores_per_worker: 12
  gpu: true
  SlurmProvider:
    partition: "gpu_rome_a100"
    account: "2022_069"
    nodes_per_block: 1
    cores_per_node: 12  
    init_blocks: 1      
    max_blocks: 1       
    walltime: "01:00:00"
    exclusive: false
    scheduler_options: "#SBATCH --clusters=dodrio\n#SBATCH --gpus=1\n"
ReferenceEvaluation:
  max_walltime: 20
  cpu_affinity: "alternating"  # avoid performance decrease in CP2K
  SlurmProvider:
    partition: "cpu_rome"
    account: "2022_069"
    nodes_per_block: 1
    cores_per_node: 64
    init_blocks: 1
    min_blocks: 0 
    max_blocks: 10 
    walltime: "01:00:00"
    exclusive: false
    scheduler_options: "#SBATCH --clusters=dodrio\n"
...
```
Each execution definition receives an additional keyword which contains all information related to the 'provider' of execution resources.
In this case, the provider is a SLURM cluster system, and 'blocks' denote individual SLURM jobs (which can run one or more workers).
The keyword-value pairs given in the `SlurmProvider` section are forwarded to the corresponding `__init__` method of the Parsl provider
([here](https://github.com/Parsl/parsl/blob/ea54919b6a85056a084e9dad9bc030806bc58fc0/parsl/providers/slurm/slurm.py#L36) for SLURM).
Check out the [configs](https://github.com/molmod/psiflow/tree/main/configs) directory for more example configurations.
since you do not need to take care that all software is installed in the same way on multiple partitions.
This setup is especially convenient to combine with containerized execution.

For people who chose to install psiflow and its dependencies [manually](installation.md#manual),
it's important to take care that all manually installed packages can be found by each of the providers.
This is possible using the `worker_init` argument of Parsl providers, which can be used to 
activate specific Python environments or execute `module load` commands.
