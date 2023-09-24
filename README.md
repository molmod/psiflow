![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)

[![Coverage Status](https://coveralls.io/repos/github/svandenhaute/psiflow/badge.svg?branch=main&service=github)](https://coveralls.io/github/svandenhaute/psiflow?branch=main)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/molmod/psiflow)](https://github.com/molmod/psiflow/releases)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1038/s41524-023-00969-x)](https://www.nature.com/articles/s41524-023-00969-x)

UDPATE (Aug 30, 2023): we're preparing v2.0.0 directly on the main branch. The current docs do not yet reflect these changes and will be updated within the next two weeks.
To adapt your current scripts, it's best to base yourself on any of the [new examples](https://github.com/molmod/psiflow/tree/main/examples) instead.

# Interatomic potentials using online learning

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
and even container orchestration systems (e.g. Kubernetes). Visit the [psiflow documentation](https://molmod.github.io/psiflow) for more details.
