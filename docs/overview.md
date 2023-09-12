Psiflow is a modular and flexible library that allows the user to
design and execute arbitrarily complex workflows based on a variety of sampling
algorithms, trainable interaction potentials, and reference levels of theory.
While all computations are orchestrated internally using Parsl, psiflow provides
elegant high-level wrappers for datasets, models, and QM singlepoint calculations
which allow the user to define sampling and learning workflows with very little effort.
These wrappers constitute the main building blocks of psiflow, and are listed
below:

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
    form, apps are just functions, and futures are the result of a function given
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
of layers (and any other network normalization metrics) using a given training dataset,
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
that the `model.train()` command will end up being executed using a GPU on SLURM cluster,
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

# run short MD simulation using some model)
metadata = walker.propagate(model=model)

```

The following fields are always present in the `metadata` object:

- `metadata.state`: `AppFuture` of a `FlowAtoms` object which represents the final state
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
of the heat bath *T*:
$$
3N\frac{T_i}{T} \sim \chi^2(3N)
$$
in which the [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution) distribution
arises because the temperature (i.e. kinetic energy) is essentially equal to the sum of
the squares of *3N* normally distributed velocity components.
Based on the inverse cumulative distribution function and a fixed *p*-value, we can
derive a threshold temperature such that:
$$
P\left[T_i > T_{\text{thres}}\right] = 1 - p
$$
For example, for a system of 100 atoms in equilibrium at 300 K, we obtain a threshold temperature of 
about 360 K for p = 10^-2^, and about 400 K for p = 10^-4^. 
If the final simulation temperature exceeds the threshold at the last step of the MD simulation
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
        data_start=trajectory,              # walker i initialized to trajectory[i]
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

Besides the dynamic walker, we also implemented an `OptimizationWalker` which
wraps around ASE's
[preconditioned L-BFGS implementation](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#preconditioned-optimizers)
; this is an efficient optimization algorithm which typically requires less steps than either conventional L-BFGS
or first-order methods such as conjugate gradient (CG).
Note that geometry optimizations in psiflow will generally
not be able to reduce the residual forces in the system below about 0.01 eV/A
because of the relatively limited precision (`float32`) of model evaluation.

## Bias potentials and enhanced sampling
In the vast majority of molecular dynamics simulations of realistic systems,
it is beneficial to modify the equilibrium Boltzmann distribution with bias potentials
or advanced sampling schemes as to increase the sampling efficiency and reduce
redundancy within the trajectory.
The [PLUMED](https://plumed.org) library provides the user with various choices of enhanced sampling
techniques; the user specifies the input parameters in a PLUMED input file
and passes it into a molecular dynamics engine (e.g. OpenMM, GROMACS, or LAMMPS).
Similarly, in psiflow, the contents of the PLUMED input file can be directly
converted into a `PlumedBias` instance in order to apply PLUMED's enhanced
sampling magic to dynamic simulations or evaluate collective variables (and
bias energy) across a dataset of atomic configurations.

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

walker = BiasedDynamicWalker(data_train[0], bias=bias, timestep=0.5)    # initialize dynamic walker with bias
state  = walker.propagate(model)                                        # performs biased MD

```
Note that the bias instance will retain the hills that were generated during walker
propagation.
Often, we want to investigate what the final bias energy looks like as a
function of the collective variable.
To facilitate this, psiflow provides the ability to evaluate `PlumedBias` objects
on `Dataset` instances using the `bias.evaluate()` method.
The returned object is a Parsl `Future` that represents an `ndarray` of shape `(nstates, 2)`.
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
state = walker.propagate(model=model)                                           

# change bias center and width
walker.bias.adjust_restraint(variable='CV', kappa=2, center=200)
state_ = walker.propagate(model)     

# if the system had enough time to equilibrate with the bias, then the following should hold
assert state.result().get_volume() < state_.result().get_volume()

```
Finally, psiflow also explicitly supports the use of numerical bias potentials
as defined on a grid:
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
force, and virial stress before they can be used during model training.
The `BaseReference` class implements the singlepoint evaluations using specific
QM software packages and levels of theory.
At the moment, psiflow only supports CP2K as the reference level of theory,
though VASP and ORCA will be added in the near future.

The main functionality of a `BaseReference` instance is provided by its
`evaluate` method, which accepts both a `Dataset` as well as a (future of a)
single `FlowAtoms` instance, and performs the single-point calculations.
Depending on which argument it receives, it returns either a future or a `Dataset`
which contain the QM energy, forces, and stress. 

```py
_, trajectory = walker.propagate(model=model, keep_trajectory=True)    # trajectory of states

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
provided by the `FlowAtoms` class:

```py
assert labeled.result().reference_status    # True, because state is successfully evaluated
print(labeled.result().reference_stdout)    # e.g. ./psiflow_internal/000/task_logs/0000/cp2k_evaluate.stdout
print(labeled.result().reference_stderr)    # e.g. ./psiflow_internal/000/task_logs/0000/cp2k_evaluate.stderr
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

<!---
## Generators
In online learning, data generation proceeds by taking an intermediate model
(and optionally, a bias potential)
and using it in a phase space sampling algorithm in order to generate
a new structure starting from some existing structure, which is then evaluated
using a reference level of theory. In psiflow terms, this 
means that a `BaseWalker` will be propagated using a `PlumedBias` and a `BaseModel`,
and the final state that is obtained will be passed to the `BaseReference`
instance after which it may be included in training/validation datasets.
```py
from ase.io import read

start = read('atoms.xyz')

walker = DynamicWalker(state, steps=100, temperature=300)
bias   = None       # or PlumedBias(plumed_input)

state = walker.propagate(model=model, bias=bias, keep_trajectory=False)
final = reference.evaluate(state)

```
However, there are few additional considerations
that come into play when generating data with imperfect interaction potentials:

- __imposing physical constraints__: interatomic potentials such as MACE or NequIP
tend to produce unphysical states when they are not yet sufficiently trained.
For example, it is sometimes possible that two atoms essentially collide
onto each other during molecular dynamics; i.e. that the interatomic distance
becomes far smaller than what is physically reasonable.
It is not desirable to waste computational
time on evaluating those states at the DFT level or including them during training,
and psiflow gives the user the ability to define __checks__ which are applied to
the sampled data in order to include or exclude samples according to some set of rules.
If the check passes, the state is evaluated using the 
reference; if not, the walker is reset and sampling is retried with a different
configuration of initial velocities.
An example of such a check is the `InteratomicDistanceCheck`,
which, as the name suggests, computes all interatomic
distances and demands that they are all larger than some minimum threshold.
Another example is the `DiscrepancyCheck`, which evaluates the sampled configuration
using a set of models, and only accepts the state if the predictions are sufficiently
different (as to avoid including redundant samples in the training data).
This approach in literature is known as query-by-committee.
- __retry handling__: even when imposing additional constraints on the sampled states,
unexpected behavior is bound to occur.
The SCF cycles in the reference evaluation may fail to converge for some particular
configuration,
a specific worker is running on faulty hardware, or a metadynamics bias potential may have become too aggressive due to 
which the force threshold is systematically exceeded.
Generators allow to specify a number of retries both for sampling and for
the reference evaluation to avoid having to restart the entire workflow when
unexpected but insignificant failures occur.

To accomodate all of this, psiflow makes use of a `Generator` class which
groups the walker and bias into a single object, along with the retry policy.
The above code block would look like this when implemented using a generator:

```py
from psiflow.generator import Generator
from psiflow.checks import InteratomicDistanceCheck

generator = Generator(
        'simple',               # name to use when logging status of this generator
        walker,                 # e.g. DynamicWalker
        bias,                   # PlumedBias or None
        nretries_sampling=2,    # walker.propagate() will be called at most thrice
        nretries_reference=0,   # reference.evaluate() will be called precisely once
        )
checks = [InteratomicDistanceCheck(threshold=0.5)]  # reject state if d(atom1, atom2) < 0.5 A for any two atoms
state = generator(model, reference, checks=checks)  # retries are handled internally

assert state.result().reference_status              # is already evaluated by the generator
```

In online learning, a common scenario is to generate data using
many different molecular dynamics simulations; all of which with more or less the same parameters but
simply initialized in a different way (either with a different seed or a different starting configuration).
Psiflow provides a simple way to _multiply_ a generator in order to obtain a list
of generators, all of which identical except for the random number seed
(and possibly the initial configuration).
```py
generators = Generator('simple', walker, bias).multiply(10) # same initial configuration, different seed
assert type(generators) == list

initial_states = Dataset.load('initial_states.xyz')         # different initial configuration, different seed
generators = Generator('simple', walker, bias).multiply(10, initialize_using=initial_states)
```
--->

## Learning algorithms
The endgame of psiflow is to allow for seamless development and scalable
execution of online learning algorithms for interatomic
potentials.
The `BaseLearning` class provides an example interface based on which such
algorithms may be implemented.
Within the space of online learning, the most trivial approach is represented
using the `SequentialLearning` class.
In sequential learning, the data generation (as performed by a set of walkers)
is interleaved with short model training steps as to update
the knowledge in the model with the states that were sampled by the walkers
and evaluated with the chosen reference level of theory.
Take a look at the following example:
```py
from psiflow.learning import SequentialLearning


data_train = Dataset.load('initial_train.xyz')
data_valid = Dataset.load('initial_valid.xyz')

walkers = DynamicWalker.multiply(     # initializes 30 walkers, with different initial configuration and seed
        30,
        data_train,                   # Dataset which provides initial configurations
        timestep=0.5,
        steps=400,
        step=50,
        start=0,
        temperature=600,
        pressure=0, # NPT
        force_threshold=30,
        initial_temperature=600,
        )

learning = SequentialLearning(              # implements sequential learning
        path_output=path_output,            # folder in which consecutive models and data should be saved
        niterations=10,                     # number of (generate, train) iterations
        train_from_scratch=True,            # whether to train with reinitialized weights in each iteration
        train_valid_split=0.9,              # partitioning of generated states into training and validation
        )

data_train, data_valid = learning.run(
        model=model,                                # initial model
        reference=reference,                        # reference level of theory
        walkers=walkers,                            # list of walkers
        )

model.save(path_output)                 # save new model separately
data_train.save('final_train.xyz')      # save final training data
data_valid.save('final_valid.xyz')      # save final validation data

```
The `learning.run()` method implements the actual online learning algorithm.
In this case, it will repeat the following
[sequence](https://github.com/molmod/psiflow/blob/master/psiflow/learning.py#L117)
of operations `niterations = 10` times:

1. deploy the model;
2. propagate each walker using the most recently deployed model, and use the
provided reference to perform the QM singlepoint evaluation of the obtained
configuration;
3. gather the configurations for which the singlepoint evaluation was successful,
and add them to any existing data;
4. reinitialize the model, and train it on the new data

After this script has executed, the `path_output` directory will contain 10
folders (named `0`, `1`, ... `9`) in which the model and datasets are logged as well
as the entire state of the walkers (i.e. start and stop configuration,
and state of the bias potentials if present).
Additional features are demonstrated in the [Examples](examples.md).
