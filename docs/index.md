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
Go to the [overview](overview.md) for a walkthrough of the most
important features. 

!!! note "Parsl: Scalable execution"
    All computations in psiflow are managed by Parsl, a scalable parallel programming
    library for Python.
    This ensures support for a large number of different execution resources,
    including clouds (e.g. Amazon Web Services, Google Cloud),
    clusters (e.g. SLURM, Torque/PBS, HTCondor)
    and even container orchestration systems (e.g. Kubernetes).
    Visit the [Parsl documentation](https://parsl.readthedocs.io/en/stable/) for more details.

<!---
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
--->


# Configuration
When executing psiflow workflows, individual training, sampling, and
QM evaluation operations are automatically organized in Parsl `apps`,
whose execution is fully customizable by the user.
For example, you could distribute all CP2K calculations to a local SLURM cluster,
perform model training on a GPU from a Google Cloud instance, and forward
the remaining phase space sampling and data processing operations to a single
workstation in your local network.
Naturally, Parsl tracks the dependencies between all objects and manages execution of the workflow
in an asynchronous manner.
Psiflow centralizes all execution-level configuration options using an `ExecutionContext`.
It forwards infrastructure-specific options within Parsl, such as the requested number of nodes
per SLURM job or the specific Google Cloud instance to be use, to training,
sampling, and QM evaluation operations to ensure they proceed as requested.
Effectively, the `ExecutionContext` hides all details of the execution
infrastructure and exposes simple and platform-agnostic resources which may be
used by training, sampling, and QM evaluation apps.
As such, we ensure that the execution-side configuration remains fully decoupled
from the computational graph itself.
For more information, check out the psiflow [Configuration](config.md) page.

!!! note "Cut some slack!"

    Psiflow is new. Its API is still subject to change, and you may
    run into unexpected behavior or undocumented features. If you do, we encourage
    you to open an issue or ask a question on the [GitHub repository](https://github.com/svandenhaute/psiflow).


[^1]: Otherwise known as active learning, incremental learning, on-the-fly learning.
