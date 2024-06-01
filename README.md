![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)


![License](https://flat.badgen.net/github/license/molmod/psiflow)
[![Docs](https://flat.badgen.net/static/docs/passing/green)](https://molmod.github.io/psiflow)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fmolmod%2Fpsiflow%2Fbadge%3Fref%3Dmain&style=flat-square)](https://actions-badge.atrox.dev/molmod/psiflow/goto?ref=main)
![Python](https://flat.badgen.net/static/python/3.10%20|%203.11/blue)
![Code style](https://flat.badgen.net/static/code%20style/black/black)
[![DOI](https://flat.badgen.net/static/DOI/10.1038%2Fs41524-023-00969-x)](https://www.nature.com/articles/s41524-023-00969-x)


**NOTE**: psiflow is still heavily in development. API is tentative, and docs are still largely TODO.

# Scalable Molecular Simulation

Psiflow is a scalable molecular simulation engine for chemistry and materials science applications.
It supports:
- **quantum mechanical calculations** at various levels of theory (GGA and hybrid DFT, post-HF methods such as MP2 or RPA, and even coupled cluster; using CP2K|GPAW|ORCA)
- **trainable interaction potentials** as well as easy-to-use universal potentials, e.g. [MACE-MP0](https://arxiv.org/abs/2401.00096)
- a wide range of **sampling algorithms**: NVE|NVT|NPT, path-integral molecular dynamics, alchemical replica exchange, metadynamics, phonon-based sampling, ...  (thanks to [i-PI](https://ipi-code.org/))

Users may define arbitrarily complex workflows and execute them **automatically** on local, HPC, and/or cloud infrastructure.
To achieve this, psiflow is built using [Parsl](https://parsl-project.org/): a parallel execution library which manages job submission and workload distribution.
As such, psiflow can orchestrate large molecular simulation pipelines on hundreds or even thousands of nodes.

# Setup

Use the following one-liner to create a lightweight [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) Python environment with all dependencies readily available:
```sh
curl -L raw.githubusercontent.com/molmod/psiflow/install.sh
```
The environment can be activated by sourcing the `activate.sh` file which has been created in the current working directory.

Next, create a `config.yaml` file which defines the compute resources. For SLURM-based HPC systems, psiflow can initialize your configuration automatically via the following command:
```sh
python -c 'import psiflow; psiflow.detect()'
```
Example configuration files for [LUMI](https://lumi-supercomputer.eu/), [MeluXina](https://luxembourg.public.lu/en/invest/innovation/meluxina-supercomputer.html), or [VSC](https://www.vscentrum.be/) can be found [here](https://github.com/molmod/psiflow/tree/main/configs).
No additional software compilation is required since all of the heavy lifting (CP2K/ORCA/GPAW, PyTorch model training, i-PI dynamics) is executed within preconfigured [Apptainer](https://apptainer.org/)/[Singularity](https://sylabs.io/singularity/) containers which are production-ready for most HPCs.

More detailed information regarding model setup and execution is coming soon.

# Examples

- [Replica exchange molecular dynamics](https://github.com/molmod/psiflow/tree/main/examples/alanine_replica_exchange) | **alanine dipeptide**: replica exchange molecular dynamics simulation of alanine dipeptide, using the MACE-MP0 universal potential.
  The inclusion of high-temperature replicas allows for fast conformational transitions and improves ergodicity.
- [Geometry optimizations](https://github.com/molmod/psiflow/tree/main/examples/formic_acid_transition) | **formic acid dimer**: approximate transition state calculation for the proton exchange reaction in a formic acid dimer,
  using simple bias potentials and a few geometry optimizations.
- [Static and dynamic frequency analysis](https://github.com/molmod/psiflow/tree/main/examples/h2_static_dynamic) | **dihydrogen**: Hessian-based estimate of the H-H bond strength and corresponding IR absorption frequency, and a comparison with a dynamical estimate from NVE simulation and Fourier analysis.
  
- [Bulk modulus calculation](https://github.com/molmod/psiflow/tree/main/examples/iron_bulk_modulus) | **iron**: estimate of the bulk modulus of fcc iron using a series of NPT simulations at different pressures
  
- [Free energy calculations](https://github.com/molmod/psiflow/tree/main/examples/iron_harmonic_fcc_bcc) | **iron**: estimating the relative stability of fcc and bcc iron with anharmonic corrections using thermodynamic integration (see e.g. [Phys Rev B., 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.054102))

- [DFT singlepoints](https://github.com/molmod/psiflow/tree/main/examples/water_cp2k_noise) | **water**: analysis of the numerical noise DFT energy and force evaluations using CP2K and the RPBE(D3) functional, for a collection of water molecules.
  
- [Path-integral molecular dynamics](https://github.com/molmod/psiflow/examples/water_path_integral_md) | **water**: demonstration of the impact of nuclear quantum effects on the variance in O-H distance in liquid water. Path-integral molecular dynamics simulations with increasing number of beads (1, 2, 4, 8, 16) approximate the proton delocalization, and lead to systematically larger variance in O-H distance.
  
- [Machine learning potential training](https://github.com/molmod/psiflow/examples/water_train_validate) | **water**: simple training and validation script for MACE on a small dataset of water configurations.
