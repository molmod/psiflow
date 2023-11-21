# Installation

Psiflow is designed as an end-to-end framework for developing interatomic potentials. As such, it has a number of dependencies
which should be available in order to be able to perform all steps in the workflow. The following table groups 
the main dependencies according to how they are used in psiflow:

<center>

| category              | name      | version   | uses GPU          | uses MPI  |
| --------------------  | --------  | -------   | :---------------: | :--------:  |
| **QM evaluation**         | CP2K      | 2023.1    |  | :material-check: |
|                       | NWChem    | 7.2       |  | :material-check: |
|                       | PySCF     | 2.4       |  | |
| **trainable potentials**  | MACE      | 0.2.0     | :material-check:  |
|                       | NequIP    | 0.5.6     | :material-check:  |
|                       | Allegro   | 0.2.0     | :material-check:  |
| **molecular dynamics**| OpenMM    | 8.0       | :material-check:  |
|                       | PLUMED    | 2.9.0     |  |
|                       | YAFF      | 1.6.0     |  |
| **miscellaneous**     | Parsl     | 2023.10.23 |  |
|                       | e3nn      | 0.4.4     | :material-check:  |
|                       | PyTorch   | 1.13.1    | :material-check:  |
|                       | ASE       | 3.22.1    |  |
|                       | Pymatgen  | 2023.11.12 |  |
|                       | wandb     | 0.15.8    |  |
|                       | Python    | 3.9       |  |

</center>

## Containerized
To alleviate users from having to go through all of the installation
shenanigans, psiflow provides a convenient portable entity which bundles all of the above
dependencies -- a container image!
Whether you're executing your calculations on a high-memory node in a cluster
or using a GPU from a Google Cloud instance, all that is required is a working
container engine and you're good to go.
The vast majority of HPCs and cloud computing providers support containerized execution,
using engines like [Apptainer/Singularity](https://apptainer.org/),
[Shifter](https://docs.nersc.gov/development/shifter/how-to-use/),
or [Docker](https://www.docker.com/).
These engines are also very easily installed on your local workstation, which facilitates
local debugging.

Besides a container engine, it's necessary to install a standalone Python environment
which needs to take care of possible job submissions and input/output writing.
Since the actual calculations are performed inside the container, the standalone
Python environment requires barely anything, and is straightforward to install
using `micromamba` -- a blazingly fast drop-in replacement for `conda`:

```console
micromamba create -n psiflow_env -c conda-forge -y python=3.9
micromamba activate psiflow_env
pip install parsl==2023.10.23 git+https://github.com/molmod/psiflow
```
That's it! Before running actual calculations, it is still necessary to set up Parsl
to use the compute resources you have at your disposal -- whether it's a local GPU,
a SLURM cluster, or a cloud computing provider; check out the
[Execution](execution.md) page for more details.

!!! note "Containers 101"

    Apptainer -- now the most widely used container system for HPCs -- is part of the
    Linux Foundation. It is easy to set up on most Linux distributions, as explained in the [Apptainer documentation](https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages).

    Psiflow's containers are hosted on the GitHub Container Registry (GHCR), for both Python 3.9 and 3.10.
    To download and run commands in them, simply execute:

    ```console
    # show available pip packages
    apptainer exec oras://ghcr.io/molmod/psiflow:3.0.0_python3.9_cuda /usr/local/bin/entry.sh pip list

    # inspect cp2k version
    apptainer exec oras://ghcr.io/molmod/psiflow:3.0.0_python3.9_cuda /usr/local/bin/entry.sh cp2k.pmsp --version
    ```

    Internally, Apptainer will store the container in a local cache directory such that it does not have to
    redownload it every time a command gets executed. Usually, it's a good idea to manually change the location 
    of these cache directories since they can end up clogging your `$HOME$` directory quite quickly.
    To do this, simply put the following lines in your `.bashrc`:

    ```console

    export APPTAINER_CACHEDIR=/some/dir/on/local/scratch/apptainer_cache
    ```

    If your compute resources use SingularityCE instead of Apptainer,
    replace 'APPTAINER' with 'SINGULARITY' in the environment variable names.

!!! note "Weights & Biases"
    To ensure psiflow can communicate its data to [W&B](https://wandb.ai), add 
        
    ```console
    export WANDB_API_KEY=<your key from wandb.ai/authorize>
    ```
    to your `.bashrc`.

!!! note "AMD GPU support"

    As the name of the container suggests, GPU acceleration for PyTorch models in OpenMM
    is currently only available for Nvidia GPUs because the compatibility of conda/mamba
    with AMD GPUs (HIP) is not great at the moment. If you really must use AMD GPUs
    in psiflow, you'll have to manually create a separate Python environment with a ROCm-enabled
    PyTorch for training, and the regular containerized setup for CPU-only
    molecular dynamics with OpenMM.

    A ROCm-compatible PyTorch can be installed using the following command:
    ```console
    pip install --force torch==1.13.1 --index-url https://download.pytorch.org/whl/rocm5.2
    ```


## Manual
While a containerized setup guarantees reproducibility and is faster to install,
a fully manual setup of Psiflow and its dependencies provides the user with full control
over software versions or compiler flags.
While this is not really necessary in the vast majority of cases, we mention for completeness
the following manual setup using `micromamba`:
```console
CONDA_OVERRIDE_CUDA="11.8" micromamba create -p ./psiflow_env -y -c conda-forge \
    python=3.9 pip ndcctools=7.6.1 \
    openmm-plumed openmm-torch pytorch=1.13.1=cuda* \
    nwchem py-plumed cp2k && \
    micromamba clean -af --yes
pip install cython==0.29.36 matscipy prettytable && \
    pip install git+https://github.com/molmod/molmod && \
    pip install git+https://github.com/molmod/yaff && \
    pip install e3nn==0.4.4
pip install numpy ase tqdm pyyaml 'torch-runstats>=0.2.0' 'torch-ema>=0.3.0' mdtraj tables
pip install git+https://github.com/acesuit/MACE.git@55f7411 && \
    pip install git+https://github.com/mir-group/nequip.git@develop --no-deps && \
    pip install git+https://github.com/mir-group/allegro --no-deps && \
    pip install git+https://github.com/svandenhaute/openmm-ml.git@triclinic
pip install git+https://github.com/molmod/psiflow
```
This is mostly a copy-paste from psiflow's [Dockerfiles](https://github.com/molmod/psiflow/blob/main/container).
