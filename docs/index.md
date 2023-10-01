---
hide:
  - toc
---

# **psiflow** - interatomic potentials using online learning

Psiflow is a modular and scalable library for developing interatomic potentials.
It interfaces popular trainable interaction potentials with
quantum chemistry software and is designed as an end-to-end framework;
it can orchestrate all computational tasks between an initial atomic structure and
the final accurate interatomic potential.
To achieve this, psiflow implements the following high-level abstractions:

- a trainable **interaction potential** (e.g. NequIP or MACE)
- one or more **phase space sampling** algorithms (e.g. biased NPT, geometry optimization)
- a reference **level of theory** (e.g. CP2K using PBE-D3(BJ) + TZVP)

These three components are used to implement **online learning** algorithms,
which essentially interleave phase space sampling with
quantum mechanical energy evaluations and model training.
In this way, the entire (relevant part of the) phase space of the system(s)
of interest may be explored and learned by the model without ever having to
perform *ab initio* molecular dynamics.

<!---
## Core functionality 

The psiflow abstractions for a reference level of theory (`BaseReference`), 
a trainable potential (`BaseModel`), and an ensemble of phase space walkers
(`Ensemble`, `BaseWalker`) are subclassed by specific implementations.
They expose the main high-level functionalities that one would intuitively
expect: A `BaseReference` can label a dataset with QM energy and forces according
to some level of theory, after which a `BaseModel` instance can be trained to it.
An `Ensemble` can use that `BaseModel` to explore the phase space of the systems
of interest (e.g. using molecular dynamics) in order to generate new
atomic configurations, which can again be labeled using `BaseReference` etc.
--->

__Scalable execution__

Psiflow workflows can be executed on large HPCs and/or cloud computing infrastructure.
The individual QM calculations, model training runs,
and/or phase space sampling calculations are efficiently executed on
hundreds or thousands of nodes thanks to
[Parsl, a parallel and asynchronous execution framework](https://parsl-project.org/).
For example, you could distribute all CP2K calculations to a local SLURM cluster,
perform model training on a GPU from a Google Cloud instance, and forward
the remaining phase space sampling and data processing operations to a single
workstation in your local network.
Naturally, Parsl tracks the dependencies between all objects and manages execution of the workflow
in an asynchronous manner.
<!---
Psiflow centralizes all execution-level configuration options using an `ExecutionContext`.
It forwards infrastructure-specific options within Parsl, such as the requested number of nodes
per SLURM job or the specific Google Cloud instance to be use, to training,
sampling, and QM evaluation operations to ensure they proceed as requested.
Effectively, the `ExecutionContext` hides all details of the execution
infrastructure and exposes simple and platform-agnostic resources which may be
used by training, sampling, and QM evaluation apps.
As such, we ensure that execution-side details are strictly separated from
the definition of the computational graph itself.
For more information, check out the psiflow [Configuration](config.md) page.
--->

!!! note "Citing psiflow"

    Psiflow is developed at the
    [Center for Molecular Modeling](https://molmod.ugent.be).
    If you use it in your research, please cite the following paper:

    Machine learning Potentials for Metal-Organic Frameworks using an
    Incremental Learning Approach,
    _Sander Vandenhaute et al._,
    [npj Computational Materials](https://www.nature.com/articles/s41524-023-00969-x),
    __9__, 19 __(2023)__


---


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


## Atomic data
In psiflow, a set of atomic configurations is represented using the `Dataset` class.
It may represent training/validation data for model development, or
a trajectory of snapshots that was generated using molecular dynamics.
A `Dataset` instance mimics the behavior of a list of ASE `Atoms` instances:
```py
from psiflow.data import Dataset


data_train  = Dataset.load('train.xyz')         # create a psiflow Dataset from a file
data_subset = data_train[:10]                   # create a new Dataset instance with the first 10 states
data_train  = data_subset + data_train[10:]     # combining two datasets is easy

data = Dataset.load('lots_of_data.xyz')
train, valid = data.shuffle().split(0.9)        # shuffle structures and partition into train/valid sets
type(train)                                     # psiflow Dataset
type(valid)                                     # psiflow Dataset

```
The main difference between a psiflow `Dataset` instance and an actual Python `list` of
`Atoms` is that a `Dataset` can represent data __that will be generated in the future__.

!!! note "Parsl 101: Apps and Futures"
    To understand what is meant by 'generating data in the future', it is necessary
    to introduce the core concepts in Parsl: apps and futures. In their simplest
    form, apps are just functions, and futures are the result of an app given
    a set of inputs. Importantly, a Future already exists before the actual calculation
    is performed. In essence, a Future _promises_ that, at some time in the future, it will
    contain the actual result of the function evaluation. Take a look at the following
    example:

    ```py
    from parsl.app.app import python_app


    @python_app # convert a regular Python function into a Parsl app
    def sum_integers(a, b):
        return a + b


    sum_future = sum_integers(3, 4) # tell Parsl to generate a future that represents the sum of integers 3 and 4
    print(sum_future)               # is an AppFuture, not an integer

    print(sum_future.result())      # now compute the actual result; this will print 7 !

    ```
    The return value of Parsl apps is not the actual result (in this case, an integer), but
    an AppFuture that will store the result of the function evaluation after it has completed.
    The main reason for doing things this way is that this allows for asynchronous execution.
    For more information, check out the [Parsl documentation](https://parsl.readthedocs.io/en/stable/).

The actual atomic configurations are stored __as a Parsl future, in an attribute of the Dataset
object__.
Actually getting the data would require the user to make a `.result()` call similar
to the trivial Parsl example above.
Let's go back to the first example and try and get the actual list of `Atoms` instances:
```py
data_train = Dataset.load('train.xyz')
atoms_list = data_train.as_list()                   # returns AppFuture

isinstance(atoms_list, list)                        # returns False! 

atoms_list.result()                                 # this is the actual list


data_train[4]                   # AppFuture representing the configuration at index 4
data_train[4].result()          # actual Atoms instance

```
If the initial XYZ file was formatted in extended XYZ format and contained the potential
energy, forces, and stress of each atomic configuration,
they are also loaded in the dataset:
```py
data_train[4].result()                      # actual Atoms instance
data_train[4].result().info['energy']       # potential energy, float
data_train[4].result().info['stress']       # virial stress, 2darray of shape (3, 3)
data_train[4].result().arrays['forces']     # forces, 2darray of shape (natoms, 3)

```
While not that important for the user, it is worth mentioning that psiflow
extends ASE's `Atoms` functionality with a few additional features, mostly
for internal convenience. Practically speaking, this does not really change anything for the user,
but we mention it for completeness.
```py
from ase import Atoms
from psiflow.data import FlowAtoms

snapshot = data_train[4].result()   # convert Future of snapshot to actual snapshot
isinstance(snapshot, Atoms)         # True; FlowAtoms subclasses Atoms
type(snapshot) == Atoms             # False; it is not actually an Atoms instance
type(snapshot) == FlowAtoms         # True

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
psiflow takes a pragmatic approach and simply monitors the temperature of the walkers.
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
Psiflow gives users the ability to put a threshold on the instantaneous temperature
of the walkers based on $\sigma_T$:
$$
T_{\text{max}}(n) = T + n \cdot \sigma_T
$$
The allowed number of standard deviations $n$ is
set using the `temperature_threshold` keyword argument of `DynamicWalker`.
If the temperature at the last step of the MD simulation exceeds this threshold
(or model evaluation yielded `NaN` or `ValueError` at any point throughout the propagation),
the walker will reset its internal state to the starting configuration
in order to make sure that subsequent propagations again start from a physically
sensible structure.

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
values = bias.evaluate(data_train, variable='CV')       # compute the collective variable 'CV' and bias energy

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
which contain the QM energy, forces, and stress. 

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
it should only contain the FORCE_EVAL section.
Additional input files which define the basis sets, pseudopotentials, and
dispersion correction parameters have to be added to the calculator after initialization.
```py
from psiflow.reference import CP2KReference


cp2k_input = with file('cp2k_input.txt', 'r') as f: f.read()
reference  = CP2KReference(cp2k_input)

# register additional input files with the following mapping
# if the corresponding keyword in the CP2K input file is X, use Y as key here:
# X: BASIS_SET_FILE_NAME    ->   Y: basis_set
# X: POTENTIAL_FILE_NAME    ->   Y: potential
# X: PARAMETER_FILE_NAME    ->   Y: dftd3
reference.add_file('basis_set', 'BASIS_MOLOPT_UZH')
reference.add_file('potential', 'POTENTIAL_UZH')
reference.add_file('dftd3', 'dftd3.dat')
```

### NWChem
For nonperiodic systems, psiflow provides an interface with [NWChem](https://nwchemgit.github.io/Home.html),
which implements a plethora of DFT and post-HF methods for both periodic and nonperiodic systems.
The `NWChemReference` class essentially wraps around the ASE calculator, and is similarly easy to use:
```py
calculator_kwargs = {
        'basis': {e: '3-21g' for e in ['H', 'C', 'O', 'N']},
        'dft': {
            'xc': 'pw91lda',
            'mult': 1,
            'convergence': {
                'energy': 1e-6,
                'density': 1e-6,
                'gradient': 1e-6,
                },
            },
        }
reference = NWChemReference(**calculator_kwargs)

```

