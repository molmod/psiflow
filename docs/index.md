# **psiflow** - interatomic potentials using online learning

Psiflow is a modular and scalable library for developing interatomic potentials.
It uses Parsl to interface popular trainable interaction potentials with
quantum chemistry software, and is designed to support computational workflows
on hundreds or thousands of nodes.
Psiflow is designed as an end-to-end framework; it can orchestrate all
computational components between an initial atomic structure and the final
trained potential.
To achieve this, psiflow implements the following high-level abstractions:

- a trainable **interaction potential** (e.g. NequIP or MACE)
- one or more **phase space sampling** algorithms (e.g. biased NPT, geometry optimization)
- a reference **level of theory** (e.g. PBE-D3(BJ) + TZVP)

These three components are used to implement **online learning**[^1] algorithms,
which essentially interleave phase space sampling with
quantum mechanical energy evaluations and model training.
In this way, the entire (relevant part of the) phase space of the system(s)
of interest may be explored and learned by the model without ever having to
perform *ab initio* molecular dynamics.

All computations in psiflow are managed by Parsl, a scalable parallel programming
library for Python.
This ensures support for a large number of different execution resources,
including clouds (e.g. Amazon Web Services, Google Cloud),
clusters (e.g. SLURM, Torque/PBS, HTCondor)
and even container orchestration systems (e.g. Kubernetes).
Visit the [Parsl documentation](https://parsl.readthedocs.io/en/stable/) for more details.


## Core functionality 

The psiflow abstractions for a reference level of theory (`BaseReference`), 
a trainable potential (`BaseModel`), and an ensemble of phase space walkers
(`Ensemble`, `BaseWalker`) are subclassed by specific implementations.
They expose the main high-level functionalities that one would intuitively
expect: A `BaseReference` can label a dataset with QM energy and forces according
to some level of theory, after which a `BaseModel` instance can be trained to it.
An `Ensemble` can use that `BaseModel` to explore the phase space of the systems
of interest (e.g. using molecular dynamics) in order to generate new
atomic configurations, which can again be labeled using `BaseReference` etc.

The following is a (simplified) excerpt that illustrates how these basic
building blocks may be used to implement a simple online
learning approach:

```py
# parameters (dataclass): defines number of iterations, number of states to sample etc.
# model (type BaseModel): represents a trainable potential
# data (type Dataset): wraps a list of atomic configurations
# ensemble (type Ensemble): defines phase space sampling
# reference (type BaseReference): defines the QM level of theory
# manager (type Manager): manages IO, wandb logging

for i in range(parameters.niterations):
    model.deploy() # performs e.g. torch.jit.compile in desired precision

    # ensemble wraps a set of phase space walkers (e.g. multiple NPT simulations)
    dataset = ensemble.sample(
            parameters.nstates, # sample this number of states
            model=model, # use current best model as potential energy surface
            )
    data = reference.evaluate(dataset) # massively parallel QM evaluation of sampled states
    data_success = data.get(indices=data.success) # some calculations may fail!
    train, valid = get_train_valid_indices(
            data_success.length(),
            self.parameters.train_valid_split,
            )
    data_train.append(data_success.get(indices=train))
    data_valid.append(data_success.get(indices=valid))

    if parameters.retrain_model_per_iteration: # recalibrate scale/shift/avg_num_neighbors
        model.reset()
        model.initialize(data_train)

    epochs = model.train(data_train, data_valid) # train model for some time

    # IO, Weights & Biases logging
    manager.save( # save data, model, and the state of the ensemble
            name=str(i),
            model=model,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            data_failed=data.get(indices=data.failed),
            )
    log = manager.log_wandb( # log using wandb for monitoring
            run_name=str(i),
            model=model,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            bias=ensemble.biases[0], # possibly None
            )
```

For example, a NequIP potential (as defined by its full `config.yaml`) is
represented using a `NequIPModel`.
Its `deploy()` and `train()` methods wrap around the deploy and training
functionalities provided in the [NequIP](https://github.com/mir-group/nequip.git) Python package.
A specific CP2K input file (including basis sets, pseudopotentials, etc)
is represented by a `CP2KReference`. Its `evaluate()` method will wrap around
the `cp2k.psmp` or `cp2k.popt` executables that are provided by CP2K,
most likely prepended with the appropriate `mpirun` command.


## Execution
The code excerpt shown above forwards individual training, sampling, and
QM evaluation operations to Parsl `apps` whose execution is fully customizable
by the user.
For example, you could distribute all CP2K calculations to a local SLURM cluster,
perform model training on a GPU from a Google Cloud instance, and forward
the remaining phase space sampling and data processing operations to a single
workstation in your local network.
Naturally, Parsl tracks the dependencies between all objects and manages execution of the workflow
in an asynchronous manner.
Psiflow centralizes the execution configuration using an `ExecutionContext`.
It forwards infrastructure-specific options within Parsl, such as the requested number of nodes
per SLURM job or the specific Google Cloud instance to be use, to training,
sampling, and QM evaluation operations to ensure they proceed as requested.
Effectively, the `ExecutionContext` hides all details of the execution
infrastructure and exposes simple and platform-agnostic resources which may be
used by training, sampling, and QM evaluation apps.
As such, we ensure that the execution-side configuration remains fully decoupled
from a logical set of operations as e.g. defined in the code block above.
For more information, check out the psiflow [Setup](setup.md) page.


[^1]: Otherwise known as active learning, incremental learning, on-the-fly learning.
