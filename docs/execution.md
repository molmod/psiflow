# Execution

Psiflow provides a convenient interface to build a
complex computational graph that consists of
QM evaluations, model training, and phase space sampling, among others.
Importantly, the execution of such graphs requires different resources
depending on the specific task at hand.
For example, QM calculations typically require nodes with a large core count (64 or even 128)
and with sufficient memory, whereas model training and evaluation require a powerful
GPU.
Because a single computer can never provide the computing power that is required to
execute such workflows, psiflow is intrinsically built to support distributed
and asynchronous execution across a large variety of resources (including most HPC and cloud infrastructure).
This means that while the entire online learning workflow is defined in a single Python script,
its execution is automatically performed on tens, hundreds, or even thousands of nodes.
The configuration of psiflow's execution is defined in a single Python file; `config.py`.
It gives the user full flexibility to customize all execution-side details, such as:

- the number of cores to use for each CP2K singlepoint evaluation, as well as the specific OpenMP/MPI parallellization settings;
- the project ID from Google Compute Engine, if applicable;
- the computational resources per SLURM job, if applicable.
This includes walltime, core count, and memory, as well as e.g. the maximum number of molecular dynamics
simulations that can be executed in parallel in a single job;
- the execution of specific `module load` or `source env/bin/activate` commands to ensure all the necessary environment variables are set.

Execution parameters such as these are defined in the configuration script;
they are strictly and deliberately kept separate
from the main Python script that defines the workflow, in line with Parsl's philosophy: _write once, run anywhere_.

This section introduces the main components of the configuration script; additional
Examples can be found on the GitHub repository, in the
[configs](https://github.com/svandenhaute/psiflow/tree/main/configs) directory.

!!! note "Parsl 103: Execution"
    It may be worthwhile to take a quick look at the
    [Parsl documentation on execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html).


## 1. Configure __how__ everything gets executed
The first part of `config.py` will determine _how_ all calculations will be performed.
This includes the number of cores to use when executing a singlepoint calculation
and the MPI/OpenMP configuration, whether to perform model inference on CPU or GPU,
the amount of walltime to reserve for training, etc.
These parameters may be set in psiflow using the `Execution` dataclasses:
`ModelEvaluationExecution`, `ModelTrainingExecution`, and `ReferenceEvaluationExecution`.
Each dataclass defines a particular aspect of the execution of a workflow (
respectively, the model inference, model training, and QM evaluation).
Aside from the execution parameters, each `Execution` also defines the
Parsl [executor](https://parsl.readthedocs.io/en/stable/userguide/execution.html#executors).
that will be set up by psiflow for executing that particular operation.
We highly recommend to ensure that each definition has its own executor
(i.e. that no two labels are the same).

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


## 2. Configure __where__ everything gets executed
The next thing to configure is _where_ all the execution resources are coming
from. Psiflow (and Parsl) need to know whether they're supposed to request
a GPU in Google Cloud or submit a SLURM jobscript to a GPU cluster, for example.
Execution resources are provided using the Parsl `ExecutionProvider` class.
The easiest option is to set psiflow to only use resources on the local
computer (i.e. your own CPU, GPU, memory, and disk space).
```py
from parsl.providers import LocalProvider


providers = {
        'default': LocalProvider(),     # resources for all pre- and post-processing
        'model': LocalProvider(),       # resources for the 'model' executor (i.e. model evaluation)
        'training': LocalProvider(),    # resources for the 'training' executor (i.e. model training)
        'reference': LocalProvider(),   # resources for the 'reference' executor (i.e. for QM singlepoints)
        }
```
The final component in the `config.py` file is a function `get_config` which
accepts a single filepath as argument and which returns the full Parsl Configuration
and the execution definitions dictionary. It offers some additional customisability
in terms of how calculations are scheduled, whether caching is enabled, and how
deal with errors during the workflow.

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
See the [Hortense](https://github.com/svandenhaute/psiflow/blob/main/configs/vsc_hortense.py)
and [Stevin](https://github.com/svandenhaute/psiflow/blob/main/configs/vsc_stevin.py)
example configurations for more details.

## 3. Putting it all together: `psiflow.load`
The execution configuration as determined by a `config.py` is to be loaded
into psiflow in order to start workflow execution.
The first step in any psiflow script is therefore to call `psiflow.load`:
it will generate a local cache directory for output
logs and intermediate files, and create a global `ExecutionContext` object.
The `BaseModel`, `BaseReference`, and `BaseWalker` subclasses
will use the information in the execution context to create and store
Parsl apps with the desired execution configuration.
End users do not need to worry about the internal organization of psiflow;
all they need to make sure is that they call `psiflow.load()` on a valid
configuration file before they commence their workflow.
The following is a trivial example in which a dataset is loaded, a simple MACE
model is trained, and the result is saved for future usage.

```py title='my_script.py'
import psiflow
from psiflow.data import Dataset
from psiflow.models import MACEModel, MACEConfig


def my_scientific_breakthrough():
    data  = Dataset.load('chemical_accuracy.xyz')   # the best dataset in the world
    data_train = data[:-10]                 # first n - 10 states are training
    data_valid = data[-10:]                 # last 10 states are validation
    model = MACEModel(MACEConfig(r_max=7))  # use default MACE parameters, but increase the cutoff to 7 A
    model.initialize(data_train)            # initialize weights, and scaling/normalization factors in layers
    model.train(data_train, data_valid)     # training magic
    model.save('./')                        # done!


if __name__ == '__main__':
    psiflow.load(
            './config.py',        # will load config.py as module and execute its get_config() method
            './psiflow_internal', # directory in which to store logs; this path should not already exist
            )
    my_scientific_breakthrough()

```
