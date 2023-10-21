![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)

[![Coverage Status](https://coveralls.io/repos/github/svandenhaute/psiflow/badge.svg?branch=main&service=github)](https://coveralls.io/github/svandenhaute/psiflow?branch=main)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/molmod/psiflow)](https://github.com/molmod/psiflow/releases)
[![pages-build-deployment](https://github.com/molmod/psiflow/actions/workflows/pages/pages-build-deployment/badge.svg)](https://molmod.github.io/psiflow)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![code style](https://img.shields.io/badge/code%20style-black-black)]()
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1038/s41524-023-00969-x)](https://www.nature.com/articles/s41524-023-00969-x)


# Interatomic potentials using online learning

Psiflow is a **modular** and **scalable** library for developing interatomic potentials. It interfaces popular trainable interaction potentials with quantum chemistry software and is designed to support computational workflows on hundreds or thousands of nodes. Psiflow is designed as an end-to-end framework; it can orchestrate all computational components between an initial atomic structure and the final trained potential. In particular, it implements a variety of **active learning** algorithms which allow for efficient exploration of the system's phase space **without requiring ab initio molecular dynamics**.

Its features include:

- active learning algorithms with enhanced sampling using PLUMED
- [Weights & Biases](wandb.ai) logging for easy monitoring and analysis
- periodic (CP2K) and nonperiodic (NWChem) systems
- efficient GPU-accelerated molecular dynamics using OpenMM
- the latest equivariant potentials such as MACE and NequIP

Execution is massively parallel and powered by [Parsl](https://parsl-project.org/), a parallel execution library which supports a variety of execution resources including clouds (e.g. Amazon Web Services, Google Cloud), clusters (e.g. SLURM, Torque/PBS, HTCondor) and even container orchestration systems (e.g. Kubernetes).
While psiflow exposes an intuitive and concise API for defining complex molecular simulation workflows in a single Python script, Parsl ensures that the execution is automatically offloaded to arbitrarily large amounts of compute resources.
Visit the [documentation](https://molmod.github.io/psiflow) for more details.
