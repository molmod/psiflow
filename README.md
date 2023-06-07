![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)

[![Coverage Status](https://coveralls.io/repos/github/svandenhaute/psiflow/badge.svg?branch=main&service=github)](https://coveralls.io/github/svandenhaute/psiflow?branch=main)

## interatomic potentials using online learning

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

These three components are used to implement online learning algorithms,
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

Visit the [psiflow documentation](https://molmod.github.io/psiflow) for more details.

## quick setup

The easiest way to get started is via the provided
[Apptainer](https://apptainer.org/)/[Singularity](https://sylabs.io/singularity/) containers.
These package all the necessary Python and non-Python dependencies (including e.g. Numpy,
GPU-enabled PyTorch, but also CP2K and PLUMED) into a single container, removing much of the
installation shenanigans that is usually associated with these packages.
In a containerized setup, all that is needed is a simple Python environment with
Parsl (including the [cctools](https://github.com/cooperative-computing-lab/cctools) module) and psiflow:

```
micromamba create -n psiflow_env ndcctools=7.5.2 -c conda-forge -y python=3.9
micromamba activate psiflow_env
pip install git+https://github.com/molmod/psiflow
```
in which we use [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as an incredibly fast
and robust drop-in replacement for the better-known `conda` package manager.
The Python environment is used to start the lightweight 'master' process which orchestrates all computations but does not do any of the heavy-lifting itself.
When individual tasks are sent to worker resources (e.g. a SLURM compute node), processes are started inside a container
and as such, all of the required software and libraries are automatically accessible.

Psiflow provides two nearly identical containers; one for AMD GPUs and one for Nvidia GPUs. They differ only in the specific version of `torch`
that was included:

- for NVIDIA GPUs and CUDA >=11.3 : `oras://ghcr.io/molmod/psiflow:1.0.0-cuda11.3`; includes `torch 1.11.0+cu113`
- for AMD GPUs and ROCm 5.2       : `oras://ghcr.io/molmod/psiflow:1.0.0-rocm5.2`; includes `torch 1.13.0+rocm5.2`

Apart from that, these containers come with CP2K 2023.1, PLUMED 2.7.2, NequIP 0.5.4, and MACE 0.1.0, in addition to a bunch
of other psiflow dependencies such as e.g. Numpy, ASE, and Pymatgen; check out the psiflow Dockerfiles for more information.
