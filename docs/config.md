# Execution

Psiflow provides a high-level interface that allows the user to build a
computational graph of operations.
The computationally nontrivial components in such graphs are typically
threefold: the QM evaluations, model training, and model inference.
On average, a single QM evaluation may require multiple tens of cores for
up to one hour of walltime, model training requires multiple hours of training on state of the art
GPUs; and even model inference (e.g. molecular dynamics) can become
expensive when long simulation times are necessary.
Because a single computer cannot provide the computing resources that are necessary to execute
such workflows, psiflow is built on top of Parsl to allow for distributed execution
across a large variety of computing resources

All execution-level configuration options are expected to be bundled
in a single `config.py` file, which is passed as a command line argument when
executing a psiflow workflow. This allows users to easily execute the same
workflow with different configuration options.

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
[Hortense](https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/gent/tier1_hortense.html)
cluster in Belgium, which has the typical SLURM/Lmod/EasyBuild setup as found
in many other European HPCs.

## 3. Putting it all together: the `ExecutionContext`
Once the `config.py` is defined, psiflow should read its contents
and create the `ExecutionContext` object that is present throughout all psiflow workflows
(see the [Overview](overview.md)).
A `context` stores both the execution definitions as well as the full Parsl
configuration.
The `BaseModel`, `BaseReference`, and `BaseWalker` subclasses
will use the information in the execution context to create and store
Parsl apps with the desired execution configuration.
This is all done automatically; psiflow users should never directly interact with the `context` object.


As explained in the [Examples](examples.md), a typical psiflow script will
use more or less the following template:

```py title="scientific_breakthrough.py"
import argparse

import psiflow


def main(context, flow_logger):
    # your psiflow workflow here
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psiflow-config', action='store')
    parser.add_argument('--name', action='store', default=None)
    args = parser.parse_args()

    context, flow_logger = psiflow.experiment.initialize(   # initialize context
            args.psiflow_config,                            # path to psiflow config.py
            args.name,                                      # run name
            )
    main(context, flow_logger)

```
which is then executed via
```sh
python scientific_breakthrough.py --psiflow-config=config.py
```

