![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)


![License](https://flat.badgen.net/github/license/molmod/psiflow)
[![Docs](https://flat.badgen.net/static/docs/passing/green)](https://molmod.github.io/psiflow)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fmolmod%2Fpsiflow%2Fbadge%3Fref%3Dmain&style=flat-square)](https://actions-badge.atrox.dev/molmod/psiflow/goto?ref=main)
![Python](https://flat.badgen.net/static/python/3.9%20|%203.10/blue)
![Code style](https://flat.badgen.net/static/code%20style/black/black)
[![DOI](https://flat.badgen.net/static/DOI/10.1038%2Fs41524-023-00969-x)](https://www.nature.com/articles/s41524-023-00969-x)


Nov 3, 2023: We're preparing v3.0.0, which introduces some breaking changes that are not yet reflected in the docs (mostly with respect to the execution configuration) -- we will fix this soon.


# Interatomic potentials using online learning

Psiflow is a **modular** and **scalable** library for developing interatomic potentials. It interfaces popular trainable interaction potentials with quantum chemistry software and is designed to support computational workflows on hundreds or thousands of nodes. Psiflow is designed as an end-to-end framework; it can orchestrate all computational components between an initial atomic structure and the final trained potential. In particular, it implements a variety of **active learning** algorithms which allow for efficient exploration of the system's phase space **without requiring ab initio molecular dynamics**.

Its features include:

- active learning algorithms with enhanced sampling using PLUMED
- [Weights & Biases](wandb.ai) logging for easy monitoring and analysis
- periodic (CP2K) and nonperiodic (PySCF, NWChem) systems
- efficient GPU-accelerated molecular dynamics using OpenMM
- the latest equivariant potentials such as MACE and NequIP

Execution is massively parallel and powered by [Parsl](https://parsl-project.org/), a parallel execution library which supports a variety of execution resources including clouds (e.g. Amazon Web Services, Google Cloud), clusters (e.g. SLURM, Torque/PBS, HTCondor) and even container orchestration systems (e.g. Kubernetes).
While psiflow exposes an intuitive and concise API for defining complex molecular simulation workflows in a single Python script, Parsl ensures that the execution is automatically offloaded to arbitrarily large amounts of compute resources.
Visit the [documentation](https://molmod.github.io/psiflow) for more details.

___

Check out this seven-minute introduction to psiflow:

<a href="https://www.youtube.com/watch?v=mQC7VomFjYQ">
  <img src="./docs/parslfest_thumbnail.png" alt="drawing" width="450"/>
</a>

which was recorded at [ParslFest 2023](https://parsl-project.org/parslfest/parslfest2023.html)
