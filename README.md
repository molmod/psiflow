![psiflow](./docs/logo_light.png#gh-light-mode-only)
![psiflow](./docs/logo_dark.png#gh-dark-mode-only)

[![Coverage Status](https://coveralls.io/repos/github/svandenhaute/psiflow/badge.svg?branch=main&service=github)](https://coveralls.io/github/svandenhaute/psiflow?branch=main)

# interatomic potentials using online learning

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


# setup

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


__NOTES:__
1. container files are several GBs in size. On most clusters, the default cache location is somwehere in your `$HOME` directory, which might not be desirable. To change this
to some other location, add the following lines to your `.bashrc`:

        export APPTAINER_CACHEDIR=/dodrio/scratch/users/vsc42527/2022_050/apptainer_cache
        export APPTAINER_TMPDIR=/dodrio/scratch/users/vsc42527/2022_050/apptainer_cache

    If your compute resources use SingularityCE instead of Apptainer, replace 'APPTAINER' with 'SINGULARITY' in the environment variable names. Lastly, to ensure psiflow can communicate its data to         [W&B](https://wandb.ai), add 
    
        export WANDB_API_KEY=<your key from wandb.ai/authorize>

    to your `.bashrc` as well.

2. If you cannot setup either Apptainer or Singularity on your compute resources, you may always choose to install all of psiflow's dependencies yourself. For most cases, a simple micromamba environment will suffice:
        
        micromamba create -p ./psiflow_env ndcctools=7.5.2 cp2k plumed -c conda-forge -y python=3.9
        micromamba activate ./psiflow_env
        pip install git+https://github.com/molmod/psiflow

        pip install cython matscipy prettytable plumed
        pip install git+https://github.com/molmod/molmod.git@f59506594b49f7a8545aef0ae6fb378e361eda80
        pip install git+https://github.com/molmod/yaff.git@422570c89e3c44b29db3714a3b8a205279f7b713
        pip install e3nn==0.4.4
        pip install git+https://github.com/mir-group/nequip.git@v0.5.6
        pip install git+https://github.com/mir-group/allegro.git --no-deps
        pip install --force git+https://git@github.com/ACEsuit/MACE.git@d520abac437648dafbec0f6e203ec720afa16cf7 --no-deps

        pip uninstall torch -y && pip install --force torch==1.11.0 --index-url https://download.pytorch.org/whl/cu113

        # dev dependencies
        pip install pytest coverage coveralls
        

## first steps
An easy way to explore psiflow's features is through a number of examples.
Before doing so, we suggest taking a look at the [psiflow documentation](https://molmod.github.io/psiflow) in order to get acquainted with the API.
Next, take a look at the example configuration files [here](https://github.com/molmod/psiflow/tree/main/configs), adapt them for the compute resources you have available,
and __test__ your configuration using the following command:
```
psiflow-test your_psiflow_config.py
```
This will tell you whether all the required libraries are available for each of the execution resources you provided, including their version numbers (and in case of PyTorch, whether it finds any available GPUs).
After this, you're all set to run any of the provided [examples](https://github.com/molmod/psiflow/tree/main/configs).
