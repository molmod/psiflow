---
hide:
  - toc
---

# **psiflow** - scalable molecular simulation


Psiflow is a scalable molecular simulation engine for chemistry and materials science applications.
It supports:

- **quantum mechanical calculations** at various levels of theory (GGA and hybrid DFT, post-HF methods such as MP2 or RPA, and even coupled cluster; using CP2K|GPAW|ORCA)

- **trainable interaction potentials** as well as easy-to-use universal potentials, e.g. [MACE-MP0](https://arxiv.org/abs/2401.00096)
- a wide range of **sampling algorithms**: NVE|NVT|NPT, path-integral molecular dynamics, alchemical replica exchange, metadynamics, phonon-based sampling, ...  (thanks to [i-PI](https://ipi-code.org/))

Users may define arbitrarily complex workflows and execute them **automatically** on local, HPC, and/or cloud infrastructure.
To achieve this, psiflow is built using [Parsl](https://parsl-project.org/): a parallel execution library which manages job submission and workload distribution.
As such, psiflow can orchestrate large molecular simulation pipelines on hundreds or even thousands of nodes.

---

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

For a complete overview of all execution options, see the [configuration](configuration.md) page.

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

!!! note "Citing psiflow"

    Psiflow is developed at the
    [Center for Molecular Modeling](https://molmod.ugent.be).
    If you use it in your research, please cite the following paper:

    Machine learning Potentials for Metal-Organic Frameworks using an
    Incremental Learning Approach,
    _Sander Vandenhaute et al._,
    [npj Computational Materials](https://www.nature.com/articles/s41524-023-00969-x),
    __9__, 19 __(2023)__



<!---
- __atomic data__: the `Dataset` class represents a list of atomic configurations.
Datasets may be labeled with energy, forces, and virial stress values obtained
based on e.g. a singlepoint QM evaluation or a trained model. Each individual
item in the list is essentially an ASE `Atoms` instance with a few additional attributes
used to store output and error logs of the QM evaluation on that
configuration.
- __trainable potentials__: the `BaseModel` class defines the interface for
trainable interaction potentials such as NequIP, Allegro, or MACE. They support
initializing and training a model based on training and validation sets,
evaluation of the model on a given test dataset, and model inference 
during molecular simulations.
- __molecular simulation__: the `BaseWalker` class defines an abstract interface
for anything that takes an initial atomic configuration as input and generates new
atomic configurations as output. This includes classical molecular dynamics 
at different external conditions (NVT, NPT), but also geometry optimizations and
even simple random perturbations/random walks.
- __bias potentials and enhanced sampling__ the `PlumedBias` class exposes
the popular PLUMED library during phase space sampling.
This allows the user to introduce bias potentials
(e.g. harmonic restraints or metadynamics) into the system
in order to increase the sampling efficiency of the walkers.
- __level of theory__: the `BaseReference` class is used to define the _target_
QM reference which the model should reproduce after training. Its main functionality
is to perform massively parallel singlepoint evaluation of a dataset of 
atomic structures using a specific level of theory and quantum chemistry package.
--->

<!---
As mentioned above, psiflow uses Parsl to orchestrate execution on arbitrarily
large amounts of computing resources (e.g. hundreds of SLURM nodes).
The configuration of these resources (cluster and partition names, environments to set up) and the execution-specific options
of individual building blocks (the number of cores to reserve for each singlepoint QM evaluation,
the floating point precision of PyTorch, the minimum walltime to reserve for training a model)
are all centralized in an `ExecutionContext`; it ensures
that the calculations that need to be performed by the building blocks
are correctly forwarded to the computational resources that the user has provided.
Check out the [Configuration](execution.md) section for more details.
In what follows, we assume that a suitable `context` has been initialized.
--->

<!---
```

## Trainable potentials
Once we know how datasets are represented, we can start defining models.
Psiflow defines an abstract `BaseModel` interface which each
particular machine learning potential should subclass.
In addition, psiflow provides configuration dataclasses for each model with
reasonable defaults.

- __NequIP__    : implemented by `NequIPModel` and `NequIPConfig`
- __Allegro__   : implemented by `AllegroModel` and `AllegroConfig`
- __MACE__      : implemented by `MACEModel` and `MACEConfig`

The `BaseModel` interface ensures that each model implements the following methods

- `initialize`: compute energy shifts and scalings as well as the average number
of neighbors (and any other network normalization metrics) using a given training dataset,
and initialize model weights.
- `train`: train the parameters of a model using two separate datasets, one for
actual training and one for validation. The current model parameters are used as
starting parameters for the training
- `evaluate`: compute the energy, force, and stress predictions on a given test dataset

The following example illustrates how `Dataset` and `BaseModel` instances can be
used to train models and evaluate errors.
```py
from psiflow.data import Dataset
from psiflow.models import NequIPModel, NequIPConfig


# setup
data_train = Dataset.load('train.xyz') # load training and validation data
data_valid = Dataset.load('valid.xyz')

config = NequIPConfig()
config.num_features = 16        # modify NequIP parameters to whatever
model = NequIPModel(config)     # create model instance

# initialize, train, deploy
model.initialize(data_train)            # this will calculate the scale/shifts, and average number of neighbors
model.train(data_train, data_valid)     # train using supplied datasets

model.save('./')        # saves initialized config and model to current working directory!

# evaluate test error
data_test       = Dataset.load('test.xyz')      # test data; contains QM reference energy/forces/stress
data_test_model = model.evaluate(data_test)     # same test data, but with predicted energy/forces/stress

errors = Dataset.get_errors(        # static method of Dataset to compute the error between two datasets
        data_test,                  
        data_test_model,                  
        properties=['forces'],      # only compute the force error
        elements=['C', 'O'],        # only include carbon or oxygen atoms
        metric='rmse',              # use RMSE instead of MAE or MAX
        ).result()                  # errors is an AppFuture, use .result() to get the actual values as ndarray

```
Note that depending on how the psiflow execution is configured,
it is perfectly possible
that the `model.train()` command will end up being executed using a GPU on a SLURM cluster,
whereas model deployment and evaluation of the test error gets
executed on your local computer.
See the psiflow [Configuration](execution.md) page for more information.

In many cases, it is generally recommended to provide these models with some estimate of the absolute energy of an isolated
atom for the specific level of theory and basis set considered (and this for each element).
Instead of having the model learn the *absolute* total energy of the system, we first subtract these atomic energies in order
to train the model on the *formation* energy of the system instead, as this generally improves the generalization performance
of the model towards unseen stoichiometries.

```py
model.add_atomic_energy('H', -13.7)     # add atomic energy of isolated hydrogen atom
model.initialize(some_training_data)

model.add_atomic_energy('O', -400)      # will raise an exception; model needs to be reinitialized first
model.reset()                           # removes current model, but keeps raw config
model.add_atomic_energy('O', -400)      # OK!
model.initialize(some_training_data)    # offsets total energy with given atomic energy values per atom

```
Whenever atomic energies are available, `BaseModel` instances will automatically offset the potential energy in a (labeled)
`Dataset` by the sum of the energies of the isolated atoms; the underlying PyTorch network is then initialized/trained
on the formation energy of the system instead.
In order to avoid artificially high energy discrepancies between models trained on the formation energy on one hand,
and reference potential energies as obtained from any `BaseReference`,
the `evaluate` method will first perform the converse operation, i.e. add the energies of the isolated atoms
to the model's prediction of the formation energy.


## Molecular simulation
Having trained a model, it is possible to explore the phase space
of a physical system in order to generate new geometries. 
Psiflow defines a `BaseWalker` interface that should be used to implement specific
phase space exploration algorithms.
Each walker implements a `propagate` method which performs the phase space sampling
using a `BaseModel` instance and returns the final state in which it 'arrived'.
Each walker has a `counter` attribute which defines the number of steps that have
elapsed between its initial structure and said returned state.

Let's illustrate this using an important example: molecular dynamics with the `DynamicWalker`.
Temperature and pressure control are implemented
by means of stochastic Langevin dynamics
because it typically dampens the correlations
as compared to deterministic time propagation methods based on extended Lagrangians (e.g. Nose-Hoover).
Propagation of a walker will return a metadata
[`namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple)
which has multiple fields, some of which are specific to the type of the walker.

```py
import numpy as np
from psiflow.sampling import DynamicWalker


walker = DynamicWalker(
        data_train[0],      # initialize walker to first configuration in dataset
        timestep=0.5,       # Verlet timestep
        steps=1000,         # number of timesteps to perform
        step=100,           # frequency with which states are sampled
        start=0,            # timestep at which sampling is started
        temperature=300,    # temperature, in kelvin
        pressure=None,      # pressure, in MPa. If None, then simulation is NVT
        seed=0,             # numpy random seed with which initial Boltzmann velocities are set
        )

# run short MD simulation using some model
metadata = walker.propagate(model=model)

```

The following fields are always present in the `metadata` object:

- `metadata.state`: `AppFuture` of an `Atoms` object which represents the final state
- `metadata.counter`: `AppFuture` of an `int` representing the total number of steps
that the walker has taken since its initialization (or most recent reset).
- `metadata.reset`: `AppFuture` of a `bool` which indicates whether the walker was reset
during or after propagation (e.g. because the temperature diverged too far from its target value).

The dynamic walker in particular has a few additional fields which might be useful:

- `metadata.temperature`: `AppFuture` of a `float` representing the average temperature
during the simulation
- `metadata.stdout`: filepath of the output log of the molecular dynamics run
- `metadata.time`: `AppFuture` of a `float` which represents the total
elapsed time during propagation.

When doing active learning, we're usually only interested in the final state of each of the walkers
and whether the average temperature remained within reasonable bounds.
In that case, the returned `metadata` object contains all the necessary information about
the propagation.
However, the actual trajectory that the walker has followed can be optionally returned as
a `Dataset`:

```py
metadata, trajectory = walker.propagate(model=model, keep_trajectory=True)
assert trajectory.length().result == (1000 / 100 + 1)   # includes initial and final state
assert np.allclose(                                     # metadata contains final state
        metadata.state.result().get_positions(),
        trajectory[-1].result().get_positions(),
        )

```

!!! note "Parsl 102: Futures are necessary"
    This example should also illustrate why exactly we would represent data using
    Futures in the first place.
    Suppose that the `walker.propagate` call is configured to run on a SLURM job
    that has a walltime of only ten minutes.
    At the time of submission, all psiflow knows is that, _at some point in the future_,
    it will receive a chunk of data that represents the trajectory of the simulation.
    It cannot yet know how many states are precisely going to be present in that
    dataset; for that we would have to actually __wait__ for the result.
    This waiting is precisely what is enforced when using `.result()` on a Future.
    For example, if we would like to find out how many states were actually generated,
    we'd use the `dataset.length()` function that returns a Future of the length
    of the dataset:
    ```py
    length = trajectory.length()    # will execute before the trajectory is actually generated
    length.result()                 # enforces Python to wait until the MD calculation has finished, and then compute the actual length
    ```
    See [this page](https://parsl.readthedocs.io/en/stable/userguide/futures.html) in the Parsl documentation
    for more information on asynchronous execution.

Successful phase space exploration is typically only possible with models that
are at least vaguely aware of what the low- and high-energy configurations of
the system look like.
If simulation temperatures are too high, simulation times are too long, or
the model is simply lacking knowledge on certain important low-energy regions
in phase space, then the simulation might explode. In practice, this means
that atoms are going to experience enormous forces, fly away, and incentivize
others to do the same.
In an online learning context, there is no point in further propagating walkers
after such unphysical events have occurred because the sampled states
are either impossible to evaluate with the given reference (e.g. due to SCF
convergence issues) or do not contain any relevant information on the atomic
interactions.
While there exist a bunch of techniques in literature in order to check for such divergences,
psiflow takes a pragmatic approach and puts a ceiling value on the allowed temperature
using the `max_excess_temperature` keyword argument.
If, at the end of a simulation, the instantaneous temperature deviates from the nominal 
heat bath temperature by more than $T_{\text{excess}}$, the simulation is considered unsafe,
and the walker is reset.
Statistical mechanics provides an exact expression for the distribution of the instantaneous
temperature of the system as a function of the number of atoms *N* and the temperature
of the heat bath *T* (hit `ctrl+R` if the math does not show up correctly):
$$
3N\frac{T_i}{T} \sim \chi^2(3N)
$$
in which the [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution) distribution
arises because the temperature (i.e. kinetic energy) is essentially equal to the sum of
the squares of *3N* normally distributed velocity components.
Its standard deviation is given by:
$$
\sigma_T = \frac{T}{\sqrt{3N}}
$$
This means that for very small systems and/or very high temperatures, the system's instantaneous
temperature is expected to deviate quite a bit from its average value.
In those cases, it's important to set the allowed excess temperature sufficiently high (e.g. 300 K)
in order to avoid resetting walkers unnecessarily.

In practical scenarios, phase space exploration is often performed in a massively
parallel manner, i.e. with multiple walkers.
The `multiply()` class method provides a convenient way of initializing a `list` of
`BaseWalker` instances which differ only in the initial starting
configuration and their random number seed.
Let us try and generate 10 walkers which are initialized with different
snapshots from the trajectory obtained before:

```py

walkers = DynamicWalker.multiply(
        10,
        data_start=trajectory,              # walker i initialized to trajectory[i];
        temperature=300,
        steps=100,
        )
for i, walker in enumerate(walkers):
    assert walker.seed == i                 # unique seed for each walker

states = []                                 # keep track of 'Future' states
for walker in walkers:
    metadata = walker.propagate(model=model)   # proceeds in parallel!
    states.append(metadata.state)
data = Dataset(states)                      # put them in a Dataset

```
If the requested number of walkers is larger than the number of states in the trajectory,
states are assigned to walkers based on their index __modulo__ the length of the trajectory.

Besides the dynamic walker, we also implemented an `OptimizationWalker` which
wraps around ASE's
[preconditioned L-BFGS implementation](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#preconditioned-optimizers)
; this is an efficient optimization algorithm which typically requires less steps than either conventional L-BFGS
or first-order methods such as conjugate gradient (CG).
Geometry optimizations in psiflow will generally
not be able to reduce the residual forces in the system below about 0.01 eV/A
because of the relatively limited precision (`float32`) of model evaluation.
Previous versions did in fact support `float64`, but since the ability to
perform precise geometry optimizations is largely irrelevant in the context of
active learning, we decided to remove this for simplicity.
Similarly, psiflow does not offer much flexibility in terms of integration algorithms because this is typically
not very important when generating atomic geometries for online learning.
The important thing is that the system has enough flexibility to explore the relevant parts of the phase space
(i.e. allowing energy and or unit cell parameters to change); how exactly this is achieved is less relevant.
For precise control of downstream inference tasks with trained models, we encourage users to employ
the PyTorch models as saved by `model.save(...)` in standalone scripts outside of psiflow.

### Bias potentials and enhanced sampling
In the vast majority of molecular dynamics simulations of realistic systems,
it is beneficial to modify the equilibrium Boltzmann distribution with bias potentials
or advanced sampling schemes as to increase the sampling efficiency and reduce
redundancy within the trajectory.
In psiflow, this is achieved by interfacing the dynamic walkers
with the [PLUMED](https://plumed.org) library, which provides the user with various choices of enhanced sampling
techniques.
This allows users to apply bias potentials along specific collective variables or evaluate the bias energy
across a dataset of atomic configurations.

!!! note "Variable names in PLUMED input files"
    For convenience, psiflow assumes that all collective variables in the PLUMED file have a name
    that starts with `CV`; `CV1`, `CV_1`, `CV_first`, or `CV` are all valid names while `cv1`, `colvar1` are not.

In the following example, we define the PLUMED input as a multi-line string in
Python. We consider the particular case of applying a metadynamics bias to
a collective variable - in this case the unit cell volume.
Because metadynamics represents a time-dependent bias,
it relies on an additional _hills_ file which keeps track of the location of
Gaussian hills that were installed in the system at various steps throughout
the simulation. Psiflow automatically takes care of such external files, and
their file path in the input string is essentially irrelevant.
To apply this bias in a simulation, we employ the `BiasedDynamicWalker`; it is
almost identical to the `DynamicWalker` except that it accepts an additional
(mandatory) `bias` keyword argument during initialization:
```py
from psiflow.sampling import BiasedDynamicWalker, PlumedBias


plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=10 LABEL=metad FILE=dummy
"""
bias = PlumedBias(plumed_input)        # a new hills file is generated

walker   = BiasedDynamicWalker(data_train[0], bias=bias, timestep=0.5)  # initialize dynamic walker with bias
metadata = walker.propagate(model)                                      # performs biased MD

```
Note that the bias instance will retain the hills that were generated during walker
propagation.
Often, we want to investigate what the final bias energy looks like as a
function of the collective variable.
To facilitate this, psiflow provides the ability to evaluate `PlumedBias` objects
on `Dataset` instances using the `bias.evaluate()` method.
The returned object is a Parsl `Future` that represents an `ndarray` of shape `(nstates, ncolvars + 1)`.
The first column represents the value of the collective variable for each state,
and the second column contains the bias energy.

```py
values = bias.evaluate(data_train)       # compute the collective variable 'CV' and bias energy

assert values.result().shape[0] == data_train.length().result()  # each snapshot is evaluated separately
assert values.result().shape[1] == 2                             # CV and bias per snapshot, in PLUMED units!
assert not np.allclose(values.result()[:, 1], 0)                 # bias energy from added hills
```
As another example, let's consider the same collective variable but now with
a harmonic bias potential applied to it.
Because sampling with and manipulation of harmonic bias potentials is ubiquitous
in free energy calculations, psiflow provides specific functionalities for this
particular case.
```py
plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
walker = BiasedDynamicWalker(data_train[0], bias=PlumedBias(plumed_input))  # walker with harmonic bias
state = walker.propagate(model=model).state

# change bias center and width
walker.bias.adjust_restraint(variable='CV', kappa=2, center=200)
state_ = walker.propagate(model).state

# if the system had enough time to equilibrate with the bias, then the following should hold
assert state.result().get_volume() < state_.result().get_volume()

```
Finally, psiflow also explicitly supports the use of bias potentials as defined numerically
on a grid of CV values:
```py
import numpy as np
from psiflow.sampling.bias import generate_external_grid

bias_function = lambda x: np.exp(-0.01 * (x - 150) ** 2)    # Gaussian hill at CV=150
grid_values   = np.linspace(0, 300, 500)                    # CV values for numerical grid

grid = generate_external_grid(      # generate contents of PLUMED grid file
        bias_function,
        grid_values,                
        'CV',                       # use ARG=CV in the EXTERNAL action
        periodic=False,             # periodicity of CV
        )
plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
EXTERNAL ARG=CV FILE=dummy
"""
bias = PlumedBias(plumed_input, data={'EXTERNAL': grid})   # pass grid file as external dependency
```
Note that metadynamics hills files cannot be shared between walkers 
(as is the case in multiple walker metadynamics) as this would
violate their strict independence.

!!! note 
    PLUMED interfacing is not supported for the `OptimizationWalker` because
    (i) it is rarely ever useful to add a bias during optimization, and (ii)
    the optimization is performed in ASE, and
    ASE's PLUMED interface is shaky at best.


## Level of theory
Atomic configurations should be labeled with the correct QM energy,
force, and (optionally) virial stress before they can be used during model training.
The `BaseReference` class implements the singlepoint evaluations using specific
QM software packages and levels of theory.
Its main functionality is provided by its
`evaluate` method, which accepts both a `Dataset` as well as a (future of a)
single `FlowAtoms` instance, and performs the single-point calculations.
Depending on which argument it receives, it returns either a future or a `Dataset`
which contain the QM energy, forces, and/or stress. 

```py
_, trajectory = walker.propagate(model=model, keep_trajectory=True)    # trajectory of states

reference = ...                           # initialize some Reference instance, e.g. CP2KReference (see below)
labeled = reference.evaluate(trajectory)  # massively parallel evaluation (returns new Dataset with results)   
assert isinstance(labeled, Dataset)
print(labeled[0].result().info['energy']) # cp2k potential energy!

labeled = reference.evaluate(trajectory[0])     # evaluates single state (returns a FlowAtoms future)
assert isinstance(labeled, AppFuture)
assert isinstance(labeled.result(), FlowAtoms)
print(labeled.result().info['energy'])          # will print the same energy
```
The output and error logs that were generated during the actual evaluation
are automatically stored in case they need to checked for errors or unexpected
behavior.
Their location in the file system is kept track of using additional attributes
provided by the `FlowAtoms` class (which is not present in ASE's `Atoms` instance):

```py
assert labeled.result().reference_status    # True, because state is successfully evaluated
print(labeled.result().reference_stdout)    # e.g. ./psiflow_internal/000/task_logs/0000/cp2k_evaluate.stdout
print(labeled.result().reference_stderr)    # e.g. ./psiflow_internal/000/task_logs/0000/cp2k_evaluate.stderr
```
Reference instances provide a convenient interface of computing the absolute energy of an isolated atom:
```py
energy_H = reference.compute_atomic_energy('H', box_size=5)
energy_H.result()   # about -13.7 eV
```

### CP2K
The `CP2KReference` expects a traditional CP2K
[input file](https://github.com/molmod/psiflow/blob/main/examples/data/cp2k_input.txt)
(again represented as a multi-line string in Python, just like the PLUMED input);
it should only contain the `FORCE_EVAL` section, and any `TOPOLOGY` or `CELL` information
will be automatically removed since this information may change from structure to structure
and is automatically taken care of by psiflow internally.
Do not use absolute filepaths to refer to basis set or pseudopotential input files.
Instead, you can simply use the corresponding filenames as they appear within the
[CP2K data directory](https://github.com/cp2k/cp2k/tree/master/data).
```py
from psiflow.reference import CP2KReference


cp2k_input = with file('cp2k_input.txt', 'r') as f: f.read()
reference  = CP2KReference(cp2k_input)

```

Sometimes, you may wish to perform energy-only evaluations. For example, in some implementations
of post-HF methods such as MP2 or RPA, evaluation of the forces can become much more expensive
and is generally not efficient.
In those cases, it is possible to perform energy-only evaluations of atomic structures, provided
that you have expressed to psiflow that it should not try to parse any forces from the output file.
This is done by providing a `properties` argument during initialization of the `Reference` instance.

```py
reference_Eonly = CP2KReference(cp2k_input, properties=('energy',))
reference_Ef    = CP2KReference(cp2k_input, properties=('energy', 'forces'))

state = reference_Eonly.evaluate(atoms)     # only contains the potential energy; not the forces
state = reference_Ef.evaluate(atoms)        # contains both energy and forces (default behavior)

```

### PySCF
The `PySCFReference` expects a string representation of the Python code that should be executed.
You may assume that this piece of code will be executed in a larger Python script in which the correct
PySCF `molecule` object has been initialized. The script should define the `energy` and `forces` Python
variables (respectively `float` and `numpy.ndarray`) which should contain the energy and negative gradient
in atomic units.

See below for an example:
```py
from psiflow.reference import PySCFReference

routine = """
from pyscf import dft

mf = dft.RKS(molecule)
mf.xc = 'pbe,pbe'

energy = mf.kernel()
forces = -mf.nuc_grad_method().kernel()
"""
basis = 'cc-pvtz'
spin = 0
reference = PySCFReference(routine, basis, spin)
```
--->
