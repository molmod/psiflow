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

This section will explain how different computing resources may be configured
with psiflow.

!!! note "Parsl 103: Execution"
    It may be worthwhile to take a quick look at the
    [Parsl documentation on execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html).


## 1. Configure __how__ everything gets executed
## 2. Configure __where__ everything gets executed
