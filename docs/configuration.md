Psiflow is easy to set up on most HPCs. The only requirement is a relatively recent
container engine:

- Apptainer >= 1.2
- SingularityCE >= 3.11

If none of these are available, contact your system administrators or set up psiflow
manually. Otherwise, proceed with the following steps.

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
  pip install git+https://github.com/molmod/psiflow.git@v4.0.0-rc0
  ```
_Everything else_ -- i-PI, CP2K, GPAW, Weights & Biases, PLUMED, ... -- is handled by
the container image and hence need not be installed manually.

- **(with `virtualenv`/`venv`)**: create a new environment and install psiflow from github
  using the same command as above. In addition, you will have to set up the `cctools`
  package manually. See the
  [documentation](https://cctools.readthedocs.io/en/stable/install/) for the appropriate
  instructions.


## Execution
Psiflow scripts are executed as a simple Python process.
Internally, it relies on Parsl to analyze the dependencies between different tasks and
execute the calculations asynchronously and as fast as possible.
To achieve this, it automatically requests the compute resources it needs during
execution.

To make this work, it is necessary to define precisely (i) how each elementary calculation
(model training, CP2K singlepoint evaluation, molecular dynamics) should proceed, and (ii)
how the required resources for those calculations should be obtained.
These additional parameters are to be specified in a separate 'configuration' `.yaml` file, which is
passed into the main Python workflow script as an argument.
In what follows, we provide an exhaustive list of all execution-side parameters along with
a few examples. We suggest to go through Parsl's [documentation on
execution](https://parsl.readthedocs.io/en/stable/userguide/execution.html) first as this
will improve your understanding of what follows.

### 1. model training
This defines how `model.train` operations are performed. Since
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
evaluated. Internally, Parsl will use that information to construct the appropriate
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
`model.train`. It has a number of special keywords:

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
      OMP_PROC_BIND: spread
    ```

### 2. model evaluation
Consider the following example:
```py
import psiflow
from psiflow.sampling import Walker, sample, replica_exchange
from psiflow.geometry import Geometry
from psiflow.hamiltonians import MACEHamiltonian


def main():
    mace = MACEHamiltonian.mace_mp0()
    start = Geometry.load('start.xyz')

    walkers = []
    for i in range(8):
        walker = Walker(mace, temperature=300)

    replica_exchange(walkers[:4], trial_frequency=100)
    walkers[-1].nbeads = 8

    outputs = sample(walkers, steps=1000, step=10)
    for i, output in enumerate(outputs):
        output.trajectory.save(f'{i}.xyz')


if __name__ == '__main__':
    with psiflow.load():
        main()

```
In this example, we use MACE-MP0 to run 8 molecular dynamics simulations in the NVT
ensemble. The first four walkers are coupled with replica exchange moves, and the last
walker is set to use PIMD with 8 replicas (or beads).
