![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)


![License](https://flat.badgen.net/github/license/molmod/psiflow)
[![Docs](https://flat.badgen.net/static/docs/passing/green)](https://molmod.github.io/psiflow)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fmolmod%2Fpsiflow%2Fbadge%3Fref%3Dmain&style=flat-square)](https://actions-badge.atrox.dev/molmod/psiflow/goto?ref=main)
![Python](https://flat.badgen.net/static/python/3.10%20|%203.11/blue)
![Code style](https://flat.badgen.net/static/code%20style/black/black)
[![DOI](https://flat.badgen.net/static/DOI/10.1038%2Fs41524-023-00969-x)](https://www.nature.com/articles/s41524-023-00969-x)


**NOTE**: we are preparing v4.0.0 -- the API is approximately final but the
documentation is still WIP. If you encounter a bug or would like to request a feature,
feel free to [open an issue](https://github.com/molmod/psiflow/issues/new)!

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
curl -L molmod.github.io/psiflow/install.sh | bash
```
The environment can be activated by sourcing the `activate.sh` file which will be created in the current working directory.
Next, create a `config.yaml` file which defines the compute resources. For SLURM-based HPC systems, psiflow can initialize your configuration automatically via the following command:
```sh
python -c 'import psiflow; psiflow.setup_slurm_config()'
```
Example configuration files for [LUMI](https://lumi-supercomputer.eu/), [MeluXina](https://luxembourg.public.lu/en/invest/innovation/meluxina-supercomputer.html), or [VSC](https://www.vscentrum.be/) can be found [here](https://github.com/molmod/psiflow/tree/main/configs).
No additional software compilation is required since all of the heavy lifting (CP2K/ORCA/GPAW, PyTorch model training, i-PI dynamics) is executed within preconfigured [Apptainer](https://apptainer.org/)/[Singularity](https://sylabs.io/singularity/) containers which are production-ready for most HPCs.

That's it! Contrary to frameworks like pyiron or aiida, psiflow does not require any databases or annoying web servers.
The only requirement is that you set up a Python environment and provide a `config.yaml`.

# Examples

- [Replica exchange molecular dynamics](https://github.com/molmod/psiflow/tree/main/examples/alanine_replica_exchange.py) | **alanine dipeptide**: replica exchange molecular dynamics simulation of alanine dipeptide, using the MACE-MP0 universal potential.
  The inclusion of high-temperature replicas allows for fast conformational transitions and improves ergodicity.
- [Geometry optimizations](https://github.com/molmod/psiflow/tree/main/examples/formic_acid_transition.py) | **formic acid dimer**: approximate transition state calculation for the proton exchange reaction in a formic acid dimer,
  using simple bias potentials and a few geometry optimizations.
- [Static and dynamic frequency analysis](https://github.com/molmod/psiflow/tree/main/examples/h2_static_dynamic.py) | **dihydrogen**: Hessian-based estimate of the H-H bond strength and corresponding IR absorption frequency, and a comparison with a dynamical estimate from NVE simulation and Fourier analysis.
  
- [Bulk modulus calculation](https://github.com/molmod/psiflow/tree/main/examples/iron_bulk_modulus.py) | **iron**: estimate of the bulk modulus of fcc iron using a series of NPT simulations at different pressures
  
- [Solid-state phase stabilities](https://github.com/molmod/psiflow/tree/main/examples/iron_harmonic_fcc_bcc.py) | **iron**: estimating the relative stability of fcc and bcc iron with anharmonic corrections using thermodynamic integration (see e.g. [Phys Rev B., 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.054102))

- [DFT singlepoints](https://github.com/molmod/psiflow/tree/main/examples/water_cp2k_noise.py) | **water**: analysis of the numerical noise DFT energy and force evaluations using CP2K and the RPBE(D3) functional, for a collection of water molecules.
  
- [Path-integral molecular dynamics](https://github.com/molmod/psiflow/examples/water_path_integral_md.py) | **water**: demonstration of the impact of nuclear quantum effects on the variance in O-H distance in liquid water. Path-integral molecular dynamics simulations with increasing number of beads (1, 2, 4, 8, 16) approximate the proton delocalization, and lead to systematically larger variance in O-H distance.
  
- [Machine learning potential training](https://github.com/molmod/psiflow/examples/water_train_validate.py) | **water**: simple training and validation script for MACE on a small dataset of water configurations.

# FAQ

**Where do I start?**
Take a brief look at the [examples](https://github.com/molmod/psiflow/examples/) or the
[documentation](https://molmod.github.io/psiflow) to get an idea for psiflow's
capabilities. Next, head over to the [setup & configuration](https://molmod.github.io/psiflow/configuration/) section of the docs to get started!

**Is psiflow a workflow manager?**
Absolutely not! Psiflow is a Python library which allows you to perform complex molecular simulations and scale them towards large numbers of compute nodes automatically.
It does not have 'fixed' workflow recipes, it does not require you to set up 'databases'
or 'server daemons'. The only thing it does is expose a concise and powerful API to
perform arbitrarily complex calculations in a highly efficiently manner.

**Is it compatible with my cluster?**
Most likely yes. Check which resource scheduling system your cluster uses (probably either
SLURM/PBSPro/SGE). If you're not sure, ask your system administrators or open an issue

**Can I use VASP with it?**
You cannot automate VASP calculations with it, but in 99% of cases there is either no need
to use VASP, or it's very easy to quickly perform the VASP part manually, outside of psiflow,
and do everything else (data generation, ML potential training, sampling) with psiflow.
Open an issue if you're not sure how to do this.

**I would like to have feature X**
Psiflow is continuously in development; if you're missing a feature feel free to open an
issue or pull request!

**I have a bug. Where is my error message and how do I solve it?**
Psiflow covers essentially all major aspects of computational molecular simulation (most
notably including the executation and parallelization), so there's bound to be some bug
once in a while. Debugging can be challenging, and we recommend to follow the following steps in
order:

1. Check the stderr/stdout of the main Python process (i.e. the `python main.py
   config.yaml` one). See if there are any clues. If it has contents which you don't
   understand, open an issue. If there's seemingly nothing there, go to step 2.
2. Check Parsl's log file. This can be found in the current working directory, under
   `psiflow_internal/parsl.log`. If it's a long file, search for any errors using `error`,
   `Error`, `ERROR`, ... If you find anything suspicious but do not know how to solve it,
   open an issue.
3. Check the output files of individual ML training, QM singlepoints, or i-PI molecular
   dynamics runs. These can be found under `psiflow_internal/000/task_logs/*`.
   Again, if you find an error but do not exactly know why it happens or how to solve it,
   feel free to open an issue. Most likely, it will be useful to other people as well
4. Check the actual 'jobscripts' that were generated and which were submitted to the
   cluster. Quite often, there can be a spelling mistake in e.g. the compute project you
   are using, or you are requesting a resource on a partition that is not available.
   These jobscripts (and there output and error) can be found under
   `psiflow_internal/000/submit_scripts/`.

**Where do these container images come from?**
They were generated using Docker based on the recipes in this repository, and were then
converted to `.sif` format using `apptainer`

**Can I run psiflow locally for small runs or debug purposes?**
Of course! If you do not provide a `config.yaml`, psiflow will just use your local
workstation for its execution. See e.g. [this](https://github.com/molmod/psiflow/blob/main/configs/threadpool.yaml) or [this](https://github.com/molmod/psiflow/blob/main/configs/wq.yaml) config used for testing.
