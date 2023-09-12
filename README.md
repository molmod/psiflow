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


# Setup

The easiest way to get started is via the provided
[Apptainer](https://apptainer.org/)/[Singularity](https://sylabs.io/singularity/) containers.
These package all the necessary Python and non-Python dependencies (including e.g. Numpy,
GPU-enabled PyTorch, but also CP2K and PLUMED) into a single container, removing much of the
installation shenanigans that is usually associated with these packages.
In a containerized setup, all that is needed is a simple Python environment with
Parsl (including the [cctools](https://github.com/cooperative-computing-lab/cctools) module) and psiflow:

```
micromamba create -n psiflow_env ndcctools=7.6.1 -c conda-forge -y python=3.9
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

- for NVIDIA GPUs and CUDA >=11.3 : `oras://ghcr.io/molmod/psiflow:2.0.0-cuda11.8`; includes `torch 1.13.1+cu112` from `conda-forge`
- for AMD GPUs and ROCm 5.2       : (not yet available)

Apart from that, these containers come with CP2K 2023.1, PLUMED 2.9, OpenMM 8.0, NequIP 0.5.6, and MACE 0.2.0, in addition to a bunch
of other psiflow dependencies such as e.g. Numpy, ASE, and Pymatgen; check out the psiflow [Dockerfile](https://github.com/molmod/psiflow/tree/main/Dockerfile) for more information.


__NOTES:__
1. container files are several GBs in size. On most clusters, the default cache location is somwehere in your `$HOME` directory, which might not be desirable. To change this
to some other location, add the following lines to your `.bashrc`:

        export APPTAINER_CACHEDIR=/some/dir/on/local/scratch/apptainer_cache
        export APPTAINER_TMPDIR=/some/dir/on/local/scratch/apptainer_cache

    If your compute resources use SingularityCE instead of Apptainer, replace 'APPTAINER' with 'SINGULARITY' in the environment variable names. Lastly, to ensure psiflow can communicate its data to         [W&B](https://wandb.ai), add 
    
        export WANDB_API_KEY=<your key from wandb.ai/authorize>

    to your `.bashrc` as well.

2. If you cannot setup either Apptainer or Singularity on your compute resources, you may always choose to install all of psiflow's dependencies yourself. For most cases, a simple micromamba environment will suffice:

        export CONDA_OVERRIDE_CUDA="11.8"
        micromamba create -n psiflow_env -c conda-forge python=3.9 pip ndcctools=7.6.1 openmm-plumed openmm-torch pytorch=1.13.1=cuda* cp2k nwchem py-plumed
        micromamba activate psiflow_env
   
        pip install cython==0.29.36 matscipy prettytable
        pip install git+https://github.com/molmod/molmod
        pip install git+https://github.com/molmod/yaff
        pip install e3nn==0.4.4
        pip install numpy ase tqdm pyyaml 'torch-runstats>=0.2.0' 'torch-ema>=0.3.0' mdtraj tables

        pip install git+https://github.com/acesuit/MACE.git@55f7411
        pip install git+https://github.com/mir-group/nequip.git@develop --no-deps
        pip install git+https://github.com/mir-group/allegro --no-deps
        pip install git+https://github.com/svandenhaute/openmm-ml.git@triclinic

        pip install git+https://github.com/molmod/psiflow
       
        # dev dependencies
        pip install pytest coverage coveralls
        

# First steps
An easy way to explore psiflow's features is through a number of examples.
Before doing so, we suggest taking a look at the [psiflow documentation](https://molmod.github.io/psiflow) in order to get acquainted with the API.
Next, take a look at the example configuration files [here](https://github.com/molmod/psiflow/tree/main/configs), adapt them for the compute resources you have available,
and __test__ your configuration using the following command:
```
psiflow-test your_psiflow_config.py
```
This will tell you whether all the required libraries are available for each of the execution resources you provided, including their version numbers (and in case of PyTorch, whether it finds any available GPUs).
After this, you're all set to run any of the provided [examples](https://github.com/molmod/psiflow/tree/main/configs).
