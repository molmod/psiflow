Psiflow makes it extremely easy to build
complex computational graphs that consist of
QM evaluations, model training, and a variety of phase space sampling algorithms, among others.
If your Python environment contains the required dependencies, you can execute any of the learning examples just like that:
```console
python zeolite_reaction.py
```
Model training, molecular dynamics, and reference evaluations will all proceed in separate subprocesses of the main
Python script.
However, this approach is not very useful because the execution of an average 'psiflow graph' requires different resources
depending on the specific task at hand.
For example, QM calculations typically require nodes with a large core count (64 or even 128)
and with sufficient memory, whereas model training and evaluation require one or more powerful
GPUs.
Because a single computer can never provide the computing power that is required to
execute such workflows, psiflow is intrinsically built to support distributed
and asynchronous execution across a large variety of resources (including most HPC and cloud infrastructure).
This means that while the entire online learning workflow is defined in a single Python script,
its execution is automatically performed on tens, hundreds, or even thousands of nodes.
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

The following sections will explain in more detail how psiflow can be configured to run on arbitrarily large resources.


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
Its default value is the following `lambda` expression (for MPICH):
```py
mpi_command = lambda x: f'mpirun -np {x} -bind-to core -rmk user -launcher fork'
```
4. __default execution__: this determines where the lightweight/administrative operations get executed.
Typically, this requires only a few cores and a few GBs of memory.

## Configuration
Let's illustrate how the execution definitions can be used to construct a psiflow configuration file.
Essentially, such files consist of a single `get_config()` method which should return a Parsl `Config` object
as well as a list of psiflow `ExecutionDefinition` instances as discussed above.
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


## Setup
The location where the above commands are executed will be referred to as the
**submission side**; this will typically be a local workstation or a login/compute node of a cluster.
Because all nontrivial calculations are forwarded to the appropriate compute
resources as specified in the configuration script (see below), the submission side does
not actually do any work, and it is therefore trivial to set up. 
All that is required is a Python environment in which Parsl and psiflow are
available (and, optionally, `ndcctools` for better scheduling).
We recommend using
[micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba)
-- a blazingly fast `conda` replacement -- to set this up:
```console
$ micromamba create -p ./psiflow_env ndcctools=7.6.1 -c conda-forge -y python=3.9
$ micromamba activate ./psiflow_env
$ pip install git+https://github.com/molmod/psiflow   # installs Parsl + dependencies
```
Setting up the **execution side** is technically more challenging because it
needs to have working installations of (parallelized) CP2K, PLUMED, and GPU-enabled PyTorch.

### Containerized
To alleviate users from having to go through all of the installation
shenanigans, psiflow provides all-inclusive containers which bundle all of its
dependencies into a portable entity --
a container image!
Whether you're executing your calculations on a high-memory node in a cluster
or using a GPU in google cloud, all that is required is a working [Docker](https://www.docker.com/)
or [Apptainer/Singularity](https://apptainer.org/) installation and you're good to go.
During task distribution, psiflow will automatically pull the relevant
container image from the
[GitHub Container Registry](https://github.com/molmod/psiflow/pkgs/container/psiflow)
and execute its tasks inside the container at approximately bare metal
performance.
Containerized execution is of course optional; users are free to provide their own environments
in which psiflow tasks need to execute in case they want full control over the specific
versions of each of the pieces of software.

### Manual

## 1. Configure __how__ everything gets executed
The first part of `config.py` will determine _how_ all calculations will be performed.
This includes the number of cores to use when executing a singlepoint calculation
and the MPI/OpenMP configuration, whether to perform model inference on CPU or GPU,
the maximum amount of compute time to allow for model training and/or inference, etc.
These parameters may be set in psiflow using the `Execution` dataclasses:
`ModelEvaluationExecution`, `ModelTrainingExecution`, and `ReferenceEvaluationExecution`.
Each dataclass defines a particular aspect of the execution of a workflow (
respectively, the model inference, model training, and QM evaluation).
Aside from the execution parameters, each `Execution` also defines the
Parsl [executor](https://parsl.readthedocs.io/en/stable/userguide/execution.html#executors)
that will be set up by psiflow for executing that particular operation.
In most cases, it is recommended to create a separate Parsl executor for each
definition.

```py
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference


model_evaluate = ModelEvaluationExecution(          # defines model inference (e.g. for walker propagation)
        executor='model',                           # a Parsl executor with label 'model' will be created
        device='cpu',                               # evaluate models on the CPU (instead of a GPU)
        ncores=1,                                   # use only one core per simulation
        dtype='float32',                            # precision when evaluating the network (32/64)
        walltime=30,                                # ensure a walltime of two minutes per dynamic walker 
        )
model_training = ModelTrainingExecution(            # model training is forced to be executed on GPU!
        executor='training',
        ncores=4,                                   # how many CPUs to use (typically #ncores / #ngpus)
        walltime=3,                                 # max walltime in minutes,
        )
reference_evaluate = ReferenceEvaluationExecution(  # defines how the reference calculations are performed
        executor='reference',
        device='cpu',                               # only CPU is supported for QM at the moment
        ncores=32,                                  # number of cores to use per calculation
        omp_num_threads=1,                          # parallelization (mpi_num_proc = ncores / omp_num_threads)
        mpi_command=lambda x: f'mpirun -np {x}',    # mpirun command; this is cluster-dependent sometimes
        cp2k_exec='cp2k.psmp',                      # specific CP2K executable; is sometimes cp2k.popt
        walltime=30, # in minutes                   # max walltime in minutes per singlepoint
        )
```
Next, we associate, for all `BaseModel` and `BaseReference` subclasses of interest,
the necessary execution definitions.

```py
definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        AllegroModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        }
```
!!! note
    Definitions can be registered in the dictionary __per model__ and __per reference__ level of theory.
    This would allow to set execution of NequIP inference on a GPU but MACE 
    inference on a CPU, for example.

Until here, we have completely specified *how* all operations within psiflow
should be executed. This part of the configuration file is therefore relatively
transferable between different configurations (at least if the hardware is
similar).

### 2. Configure __where__ everything gets executed
The second part of the configuration file specifies _where_ all the execution resources are coming
from. Psiflow (and Parsl) need to know whether they're supposed to request
a GPU in Google Cloud or submit a SLURM jobscript to a GPU cluster, for example.
In particular, this means that the executor labels (stored as attributes in the
execution definitions from the first part) need to be associated with
particular **resource providers**. Resource providers can be anything, from
your local workstation, the cluster at your university, to an array of VM instances in
Google Cloud.
These resources can be defined in the configuration file using any of the
following Parsl `ExecutionProvider` subclasses:

- `AWSProvider`
- `CobaltProvider`
- `CondorProvider`
- `GoogleCloudProvider`
- `GridEngineProvider`
- `LocalProvider`
- `LSFProvider`
- `GridEngineProvider`
- `SlurmProvider`
- `TorqueProvider`
- `KubernetesProvider`
- `PBSProProvider`

For each of the executor labels used in the first part, we need to define which
provider that executor should use.
The easiest option is to set psiflow to only use resources on the local
computer (i.e. your own CPU, GPU, memory, and disk space) using the
`LocalProvider`:
```py
from parsl.providers import LocalProvider


providers = {
        'default': LocalProvider(),     # resources for all pre- and post-processing
        'model': LocalProvider(),       # resources for the 'model' executor (i.e. model evaluation)
        'training': LocalProvider(),    # resources for the 'training' executor (i.e. model training)
        'reference': LocalProvider(),   # resources for the 'reference' executor (i.e. for QM singlepoints)
        }
```
Note that in addition to `model`, `training`, and `reference`, it is also
necessary to define a (simple) provider for the `default` executor,
which takes care of all administrative tasks (copying data, reading and writing XYZ files, ... ).

Once each executor label is assigned to a particular provider of your choice,
the `get_config` function can be used to combine the definitions and providers
into a [fully-fledged Parsl `Config`](https://parsl.readthedocs.io/en/stable/userguide/configuring.html).
It offers some additional customizability
in terms of how calculations are scheduled, whether caching is enabled, and how
to deal with errors during the workflow.

```py
from psiflow.execution import generate_parsl_config


def get_config(path_parsl_internal):
    config = generate_parsl_config(     # psiflow internal method to parse definitions and providers
            path_parsl_internal,        # directory in which psiflow and parsl may cache intermediate files
            definitions,
            providers,
            use_work_queue=True,        # Parsl-specific parameter which specifies the Executor class
            )
    return config, definitions
```
Example configurations for local execution can be found on the GitHub repository.
In the same directory, you'll find configuration files for the
[Flemish supercomputers](https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/gent/tier1_hortense.html)
in Belgium, which have the typical SLURM/Lmod/EasyBuild setup as found
in many other European HPCs.
Naturally, these configurations rely on one or more `SlurmProvider` instances
which provide the computational resources,
as opposed to the `LocalProvider` shown here. A `SlurmProvider` may
be configured in terms of the minimum and maximum number of jobs it may request
during the workflow,
the number of cores, nodes, GPUs, and amount of walltime per job, as well as
the cluster(s), partition(s), and account(s) to use.
See the [Hortense](https://github.com/molmod/psiflow/blob/main/configs/vsc_hortense.py)
and [Stevin](https://github.com/molmod/psiflow/blob/main/configs/vsc_stevin.py)
example configurations for more details.

### 3. Putting it all together: `psiflow.load`
The execution configuration as determined by a `config.py` is to be loaded
into psiflow in order to start workflow execution.
The first step in any psiflow script is therefore to call `psiflow.load`:
it will generate a local cache directory for output
logs and intermediate files, and create a global `ExecutionContext` object.
The `BaseModel`, `BaseReference`, and `BaseWalker` subclasses
will use the information in the execution context to create and store
Parsl apps with the desired execution configuration.
End users do not need to worry about the internal organization of psiflow;
all they need to make sure is that they provide a valid configuration
file when calling the main script and begin with a `psiflow.load()` call.
Internally, `psiflow.load()` will parse the command line arguments, read
the configuration script, and load the Parsl `Config` instance such that
the workflow can begin.

```console
    $ python my_workflow.py --psiflow-config lumi.py
```



```py title='my_workflow.py'
import psiflow


def my_scientific_breakthrough():
    ...


if __name__ == '__main__':
    psiflow.load()
    my_scientific_breakthrough()
```

