Psiflow is easy to set up on most HPCs. The only requirement is a relatively recent
container engine:

- Apptainer >= 1.2
- SingularityCE >= 3.11

To detect which of these is available on your HPC, execute `apptainer --version` or
`singularity --version` in a shell on a login or compute node. Note that on some systems, the container runtime is packaged in a
module in which case you would first have to load it before it becomes available in your
shell. Check your HPC's documentation for more information.
If none of these are available, contact your system administrators or set up psiflow
[manually](#manual-setup). Otherwise, proceed with the following steps.

We provide two versions of essentially the same container; one for Nvidia GPUs (based on a
PyTorch wheel for CUDA 11.8) and one for AMD GPUs (based on a PyTorch wheel for ROCm 5.6).
These images are hosted on the Github Container Registry (abbreviated by `ghcr`) and can
be directly downloaded and cached by the container runtime.
For example, if we wish to execute a simple command `ls` using the container image for
Nvidia GPUs, we would write:
```bash
apptainer exec oras://ghcr/io/molmod/psiflow:main_cu118 ls
```
We use `psiflow:main_cu118` to get the image which was built from the latest `main` branch
of the psiflow repository, for CUDA 11.8.
Similarly, for AMD GPUs and, for example, psiflow v4.0.0-rc1, we would use
```bash
apptainer exec oras://ghcr.io/molmod/psiflow:4.0.0-rc1_rocm5.6 ls
```
See the [Apptainer](https://apptainer.org/docs/user/latest/)/[SingularityCE](https://docs.sylabs.io/guides/4.1/user-guide/) documentation for more information.

## Python environment
The main Python script which defines the workflow requires a Python 3.10 / 3.11 environment
with a recent version of `pip` and [`ndcctools`](https://github.com/cooperative-computing-lab/cctools).

- **(without existing conda/mamba binary)**: if you do not know how to set this up yourself, use the following one-line install command:
  ```sh
  curl -L molmod.github.io/psiflow/install.sh | bash
  ```
  This command sets up `micromamba` (i.e. `conda` but 1000x faster) in a fully local
  manner, without messing up your `.bashrc` or `.zshrc` file. In addition, it creates a
  minimal Python environment with all the required packages installed. Activate the
  environment by sourcing the `activate.sh` file which will have been created in the
  current working directory:
  ```sh
  source activate.sh
  ```

- **(with existing conda binary)**: create a new 3.10/3.11 environment and make sure `pip`
  and `ndcctools` are available:
  ```
  micromamba create -n psiflow_env -y python=3.10 pip ndcctools=7.11.1 -c conda-forge
  ```
  Next, activate the environment and install psiflow from its repository:
  ```
  micromamba activate psiflow_env
  pip install git+https://github.com/molmod/psiflow.git@v4.0.0-rc1
  ```
_Everything else_ -- i-PI, CP2K, GPAW, Weights & Biases, PLUMED, ... -- is handled by
the container images and hence need not be installed manually.

- **(with `virtualenv`/`venv`)**: create a new environment and install psiflow from github
  using the same command as above. In addition, you will have to compile and install the `cctools`
  package manually. See the
  [documentation](https://cctools.readthedocs.io/en/stable/install/) for the appropriate
  instructions.

Verify the correctness of your environment using the following commands:

```bash
python -c 'import psiflow'  # python import should work
which work_queue_worker     # tests whether ndcctools is available and on PATH

```


## Execution
Psiflow scripts are executed as a simple Python process.
Internally, it relies on Parsl to analyze the dependencies between different tasks and
execute the calculations asynchronously and as fast as possible.
To achieve this, it automatically requests the compute resources it needs during
execution.

To make this work, it is necessary to define precisely how ML potential training, molecular
dynamics, and QM calculations should proceed, and (ii)
how the required resources for those calculations should be obtained.
These additional parameters are to be specified in a separate 'configuration' `.yaml` file, which is
passed into the main Python workflow script as an argument.
The configuration file has a specific structure which is explained in the following
sections. In many cases, you will be able to start from one of the [example
configurations](https://github.com/molmod/psiflow/tree/main/configs)
in the repository and adapt it for your cluster.
We also suggest you  to go through Parsl's [documentation on
execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html) first as this
will improve your understanding of what follows.

There are three types of calculations:

- **ML potential training** (`ModelTraining`)
- **ML potential inference, i.e. molecular dynamics** (`ModelEvaluation`)
- **QM calculations** (`CP2K`, `GPAW`, `ORCA`)

and the structure of a typical `config.yaml` consequently looks like this
```yaml
# top level options define the overall behavior
# see below for a full list
container_engine: <singularity or apptainer>
container_uri: <link to container, i.e. oras://ghcr.io/...>

ModelTraining:
  # specifies how ML potential training should be performed
  # and which resources it needs to use

ModelEvaluation:
  # specifies how MD / geometry optimization / hamiltonian computations are performed
  # and which resources it needs to use

CP2K:
  # specifies how CP2K single points need to be performed

GPAW:
  # specifies how GPAW single points need to be performed

ORCA:
  # specifies how ORCA single points need to be performed
  
```


### 1. ML potential training
This defines how `model.train()` operations are performed. Since
training is necessarily performed on a GPU, it is necessary to specify resources in
which a GPU is available. Consider the following simple training example
```py
import psiflow
from psiflow.models import load_model
from psiflow.data import Dataset


def main(fraction):
    model = load_model('my_previous_model')
    train, valid = Dataset.load('data.xyz').split(fraction, shuffle=True)
    model.train(train, valid)
    model.save('fraction_{}'.format(fraction))


if __name__ == '__main__':
    with psiflow.load():    # ensures script waits until everything completes
        main(0.5)
        main(0.7)
        main(0.9)

```
It will execute three independent training runs, whereby the only difference is in the
fraction of training and validation data. Importantly, because their is no dependency
between each individual run, they will automatically be executed in parallel.
Suppose we are in a terminal on e.g. a login or compute node of a SLURM cluster.
Then we can execute the script using the following command:
```sh
python train.py config.yaml
```
The `config.yaml` file should define how and where the model should be trained and
evaluated.

Next, we define how model training should be performed.
Internally, Parsl will use that information to construct the appropriate
SLURM jobscripts, send them to the scheduler, and once the resources are allocated,
start the calculation. For example, assume that the GPU partition on this cluster is
named `infinite_a100`, and it has 12 cores per GPU. Consider the following config
```yaml
ModelTraining:
  cores_per_worker: 12
  gpu: true
  slurm:
    partition: "infinite_a100"
    account: "112358"
    nodes_per_block: 1
    cores_per_node: 24
    max_blocks: 1
    walltime: "12:00:00"
    scheduler_options: "#SBATCH --gpus=2"
```
The top-level keyword `ModelTraining` indicates that we're defining the execution of
`model.train()`. It has a number of special keywords:

  - **cores_per_worker** (int): number of CPUs per GPU.
  - **gpu** (bool): whether to use GPU(s) -- should almost always be true for training.
  - **slurm** (dict): defines compute resources specifically for a SLURM scheduler (i.e.
    using `sbatch`/`salloc` commands). It should include all parameters which are
    required to create the actual jobscript. Note that the requested resources for a single slurm job
    can be larger than the required resources per worker.
    In this particular example, we ask for a single allocation of 2 GPUs and 24 cores in
    total, with a walltime of 12 hours, and with a limit on the number of running jobs of
    this type equal to 1 (`max_blocks`).

When we execute `python train.py config.yaml`, psiflow will analyze the script and
realize it needs to execute three training runs. Because we use `cores_per_worker:
12`, it sees that it can fit all three training runs on two SLURM jobs, each with two GPUs.
Of course, because `max_blocks: 1`, it will only submit one job and start two training
runs. The third training run will start only when any of the first two have finished running,
he will not submit a second SLURM job. The `max_blocks` setting is useful in scenarios
where your cluster imposes tight limits on the number of running jobs per user, or when
the queue time is long.

There exist a few additional keywords for `ModelTraining` which might be useful:

  - **max_training_time** (float, in minutes): This is the maximum time that any
    single training run can take. After this time, a `SIGTERM` is sent to the training
    process which ensures the training is gracefully interrupted and output models
    are saved. Sometimes, it is more convenient to use this rather than MACE's built
    in `max_num_epochs` parameter.
  - **env_vars** (dict): additional environment variables which might be necessary to
    achieve optimal training performance. For example, on some clusters, it is
    necessary to tune the process/thread affinity a little bit. For example:
    ```yaml
    env_vars:
      OMP_PROC_BIND: "spread"
    ```

### 2. molecular dynamics
Consider the following example:
```py
import psiflow
from psiflow.sampling import Walker, sample, replica_exchange
from psiflow.geometry import Geometry
from psiflow.hamiltonians import MACEHamiltonian


def main():
    mace = MACEHamiltonian.mace_mp0()
    start = Geometry.load('start.xyz')

    walkers = Walker(mace, temperature=300).multiply(8)

    outputs = sample(walkers, steps=int(1e9), step=10)   # extremely long
    for i, output in enumerate(outputs):
        output.trajectory.save(f'{i}.xyz')


if __name__ == '__main__':
    with psiflow.load():
        main()

```
In this example, we use MACE-MP0 to run 8 molecular dynamics simulations in the NVT
ensemble. Since they are all independent from each other, psiflow will attempt to execute
them in parallel as much as possible.
The configuration section which deals with ML potential inference, including molecular
dynamics but also geometry optimization and `hamiltonian.compute()` calls, is named
`ModelEvaluation`:


```yaml
ModelEvaluation:
  cores_per_worker: 12
  gpu: true
  slurm:
    partition: "infinite_a100"
    account: "112358"
    nodes_per_block: 2
    cores_per_node: 48          # full node; sometimes granted faster than partials
    max_blocks: 1
    walltime: "01:00:00"        # small to try and skip the queue
    scheduler_options: "#SBATCH --gpus=4"
```
It is in general quite similar to `ModelTraining`. Because in general, psiflow workflows
contain a large number of molecular dynamics simulations, it makes sense to ask for larger
allocations for each block (= SLURM job). In this example, we immediately ask for two full
GPU nodes, with four GPUs each. This is exactly the amount we need to execute all eight
molecular dynamics simulations in parallel, without wasting any resources.
As such, when we execute the above example using `python script.py config.yaml`, Parsl
will recognize that we need resources for eight simulations, ask for precisely one allocation
according to the above parameters, and start all eight simulations simultaneously.

Of course, we greatly overestimate the number of steps we wish to simulate.
The SLURM allocation has a walltime of one hour, which means that if a simulation does not
finish in 12 hours, it will be gracefully terminated and the saved trajectories will only
cover a fraction of the requested one billion steps.
Psiflow will not automatically continue the simulations on a new SLURM allocation.

The available keywords in the `ModelEvaluation` section are the same as for
`ModelTraining`, except for one:

- **max_simulation_time** (float, in minutes): 

### 3. QM calculations
Finally, we need to specify how QM calculations are performed.
By default, these calculations are not executed within the container image provided by
`container_uri` at the top level.
Users can choose to rely on their system-installed QM software or employ one of the
smaller and specialized container images for CP2K or GPAW. We will discuss both cases
below.

First, assume we wish to use a system-installed CP2K module, and execute each singlepoint
on 32 cores. Assume that the nodes in our cpu partition possess 128 cores:
```yaml
CP2K:
  cores_per_worker: 32
  max_evaluation_time: 30       # kill calculation after 30 mins; SCF unconverged
  launch_command: "OMP_NUM_THREADS=1 mpirun -np 32 cp2k.psmp"  # force 1 thread/rank
  slurm:
    partition: "infinite_CPU"
    account: "112358"
    nodes_per_block: 16
    cores_per_node: 128
    max_blocks: 1
    walltime: "12:00:00"
    worker_init: "ml CP2K/2024.1"  # activate CP2K module in jobscript!
```
We asked for a big allocation of 16 nodes, each with 128 cores. On each node, psiflow can
concurrently execute four singlepoints, since we specified `cores_per_worker: 32`.

Consider now the following script:
```py
import psiflow
from psiflow.data import Dataset
from psiflow.reference import CP2K


def main():
    unlabeled = Dataset.load('long_trajectory.xyz')

    with open('cp2k_input.txt', 'r') as f:
        cp2k_input = f.read()
    cp2k = CP2K(cp2k_input)

    labeled = unlabeled.evaluate(cp2k)
    labeled.save('labeled.xyz')


if __name__ == '__main__':
    with psiflow.load():
        main()

```
Assume `long_trajectory.xyz` is a large XYZ file with, say, 1,000 snapshots.
In the above script, we simply load the data, evaluate the energy and forces of each
snapshot with CP2K, and save the result as (ext)XYZ.
Again, we execute this script by running `python script.py config.yaml` within a Python
environment with psiflow and cctools available.
Even though all of these calculations can proceed in parallel, we specified `max_blocks:
1` to not overload our resource usage.
As such, Parsl will request precisely one block/allocation of 16 nodes, and start
executing the singlepoint QM evaluations.
At any given moment, there will be (16 nodes x 4 calculations/node = ) 64 calculations
running.

Now assume our system administrators did not provide us with the latest and greatest
version of CP2K.
The installation process is quite long and tedious (even via tools like EasyBuild or Spack),
which is why psiflow provides **small containers which only contain the QM software**.
They are separate from the psiflow containers mentioned before in order to improve
modularity and reduce individual container sizes.
At the moment, such containers are available for CP2K 2024.1 and GPAW 24.1.
To use them, it suffices to wrap the launch command inside an `apptainer` or `singularity`
invocation, whichever is available on your system:

```yaml
CP2K:
  cores_per_worker: 32
  max_evaluation_time: 30       # kill calculation after 30 mins; SCF unconverged
  launch_command: "apptainer exec -e --no-init oras://ghcr.io/molmod/cp2k:2024.1 /opt/entry.sh mpirun -np 32 cp2k.psmp"
  slurm:
    partition: "infinite_CPU"
    account: "112358"
    nodes_per_block: 16
    cores_per_node: 128
    max_blocks: 1
    walltime: "12:00:00"
    # no more need for module load commands!
```
The command is quite long but normally self-explanatory if you're somewhat familiar with
containers.

## SLURM quickstart

Psiflow contains a small script which detects the available SLURM partitions and their
hardware and creates a minimal, initial `config.yaml` which you can use as a starting point
to further tune to your liking. To use it, simply activate your psiflow Python environment 
and execute the following command:

```sh
python -c 'import psiflow; psiflow.setup_slurm_config()'
```

## manual setup
TODO
