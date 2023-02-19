Psiflow is designed as a modular and flexible library that allows the user to
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
- __bias potentials and enhanced sampling__ the `PlumedBias` class interfaces
the popular PLUMED library with specific walkers. This allows the user to
define bias potentials (e.g. harmonic restraints or metadynamics) that should
be used when a walker is propagating through phase space.
- __generators__: the `Generator` class wraps the generation of a single
atomic configuration using a walker and, optionally, a bias potential.
It implements additional features related to error handling during either the
sampling or the reference evaluation, and allows to user to include additional
checks to filter out unwanted data.
For example, newly initialized models may induce walkers to explore unphysical
regions in phase space when sampling at high temperature and/or for long
timescales.
When this happens, users might want to impose an `InteratomicDistanceCheck`
in order to filter out sampled configurations in which some of the
interatomic distances are unphysically close (e.g closer than 0.5 A).
Checks can also be employed to implement uncertainty-based data selection such
as query-by-committee.
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
Check out the [Configuration](config.md) section for more details.
In what follows, we assume that a suitable `context` has been initialized.
--->


## Atomic data
In psiflow, a set of atomic configurations is represented using the `Dataset` class.
It may represent training/validation data for model development, or
a trajectory of snapshots that was generated using molecular dynamics.
A `Dataset` instance mimics the behavior of a list of ASE `Atoms` instances:
```py
from psiflow.data import Dataset


data_train  = Dataset.load(context, 'train.xyz')    # create a psiflow Dataset from a file
data_subset = data_train[:10]                       # create a new Dataset instance with the first 10 states
data_train  = data_subset + data_train[10:]         # combining two datasets is easy

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
data_train  = Dataset.load(context, 'train.xyz')
atoms_list  = data_train.as_list()                  # returns AppFuture

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
data_train = Dataset.load(context, 'train.xyz') # load training and validation data
data_valid = Dataset.load(context, 'valid.xyz')

config = NequIPConfig()
config.num_features = 16                # modify NequIP parameters to whatever
model = NequIPModel(context, config)    # create model instance


# initialize, train, deploy
model.initialize(data_train)            # this will calculate the scale/shifts, and average number of neighbors
model.train(data_train, data_valid)     # train using supplied datasets

model.save('./')                # save initialized config, undeployed model to current working directory!
model.deploy()                  # prepare for inference, e.g. test error evaluation or molecular dynamics


# evaluate test error
data_test       = Dataset.load(context, 'test.xyz')   # test data; contains QM reference energy/forces/stress
data_test_model = model.evaluate(data_test)           # same test data, but with predicted energ/forces/stress

errors = Dataset.get_errors(        # static method of Dataset to compute the error between two datasets
        data_test,                  
        data_test_model,                  
        properties=['forces'],      # only compute the force error
        elements=['C', 'O'],        # only include carbon or oxygen atoms
        metric='rmse',              # use RMSE instead of MAE or MAX
        ).result()                  # errors is an AppFuture, use .result() to get the actual values!

```
Note that depending on the specific configurations in the `ExecutionContext` instance,
it is perfectly possible
that the `model.train()` command will end up being executed using a GPU on SLURM cluster,
whereas model deployment and evaluation of the test error gets
executed on your local computer.


## Molecular simulation
Having trained and deployed a model, it is possible to explore the phase space
of a physical system in order to generate new (and physically relevant) structures. 
Psiflow defines a `BaseWalker` interface that should be used to implement specific
phase space exploration algorithms.
Prominent examples are molecular dynamics (both NVT and NPT), as implemented
using the `DynamicWalker` class, and geometry optimizations, as implemented using
the `OptimizationWalker` class.

Molecular dynamics simulations are performed using YAFF, a simple molecular mechanics
library written in Python. It supports a variety of
[temperature](https://github.com/molmod/yaff/blob/master/yaff/sampling/nvt.py) and
[pressure](https://github.com/molmod/yaff/blob/master/yaff/sampling/npt.py)
control algorithms.
By default, psiflow employs stochastic Langevin methods for both temperature
and pressure control because they typically exhibit a smaller correlation time
as compared to deterministic methods such as NHC or MTK.
Geometry optimization is performed using the
[preconditioned L-BFGS implementation](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#preconditioned-optimizers)
in ASE, as it typically requires less steps than either conventional L-BFGS or
first-order methods such as CG.
Note that geometry optimizations require accurate force evaluations and are
therefore always performed in double precision (`float64`).
Because `model.deploy()` calls in psiflow will deploy both a single and a double
precision variant of each model, the user will not notice any of this.

Let us illustrate how the deployed model from the previous example may be used
to explore the phase of the system under study:
```py
import numpy as np
from psiflow.sampling import DynamicWalker


walker = DynamicWalker(
        context,
        data_train[0],      # initialize walker to first configuration in dataset
        timestep=0.5,       # Verlet timestep
        steps=1000,         # number of timesteps to perform
        step=100,           # frequency with which states are sampled
        start=0,            # timestep at which sampling is started
        temperature=300,    # temperature, in kelvin
        pressure=None,      # pressure, in MPa. If None, then simulation is NVT
        seed=0,             # numpy random seed with which initial Boltzmann velocities are set
        )

state, trajectory = walker.propagate(   # actual MD
        model=model,                    # use this model to evaluate forces at every step
        keep_trajectory=True,           # keep the trajectory, and return it as a Dataset
        )

```
The state that is returned by the propagation is a Future of an ASE `Atoms`
that represents the final state of the walker after the simulation.
If the simulation proceeds without errors, the following assertion
will evaluate to True:
```
assert np.allclose(
        state.result().positions,           # state is a Future; use .result() to get actual Atoms
        trajectory[-1].result().positions,  # trajectory is a Dataset; use regular indexing and .result()
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
Catching such events is nontrivial, and a few mechanisms
are in place in psiflow.
Each `BaseWalker` instance is given a tag that can only assume the values
_safe_ or _unsafe_. Dynamic walkers will be tagged as unsafe if one of the forces
in the system exceeds a certain threshold (40 eV/A by default), as this typically
indicates an upcoming explosion.
If walkers are tagged as _unsafe_, the `state` that is returned after propagation
may not be physically relevant, and it may be advised to not include those in
training or validation sets. 


## Ensemble
In online learning applications, it is almost always beneficial to maximally
parallelize the phase space sampling computations in the data generation stage.
For example, to generate 10 decorrelated configurations, it is much more efficient
to initialize 10 walkers and use `walker.propagate()` exactly once for each
walker, rather than using a single walker that performs 10 sequential propagations.
To accomodate for this scenario, psiflow provides the `Ensemble` class.
It basically contains a list of walkers, and provides a `sample()` method
which uses those walkers to sample a user-defined number of states and store
them in a `Dataset` instance.
The ensemble does not have to be homogeneous, i.e. it can contain multiple
different types of walkers.
```py
from psiflow.sampling import RandomWalker, DynamicWalker
from psiflow.ensemble import Ensemble


walkers = [
        RandomWalker(),
        RandomWalker(),
        DynamicWalker(temperature=400),
        DynamicWalker(temperature=600),
        ]
ensemble = Ensemble(context, walkers)
small = ensemble.sample(4, model)           # sample 4 states; 1 per model
large = ensemble.sample(7, model)           # sample 7 states; at least 1 per model

small.length().result()       # returns 4
large.length().result()       # returns 7

```
In online learning, a common scenario is to generate data using
many different molecular dynamics simulations; all of which with the same parameters but
simply initialized in a different way (either with a different seed or a different starting configuration).
Psiflow provides a simple way to generate an ensemble of walkers based on a
single 'template' walker; the only difference between each pair of walkers will
be the random number seed and possibly the starting configuration.
```py
template = DynamicWalker(steps=1000, temperature=600)
ensemble = Ensemble.from_walker(
        template,
        nwalkers=10,
        dataset=None,       # if dataset is not None, it provides the initial configurations for each walker
        )

print(len(ensemble.walkers))                   # 10
print(ensemble.walkers[0].parameters.seed)     # unique seed: 0
print(ensemble.walkers[1].parameters.seed)     # unique seed: 1
```

## Bias potentials and enhanced sampling
In the vast majority of molecular dynamics simulations of realistic systems,
it is beneficial to modify the equilibrium Boltzmann distribution with bias potentials
or advanced sampling schemes as to increase the sampling efficiency and reduce
redundancy within the trajectory.
The [PLUMED](plumed.org) library provides the user with various choices of enhanced sampling
techniques, and psiflow provides a specific `PlumedBias` class to implement
these techniques in the existing `DynamicWalker` implementation of molecular
dynamics.

In the following example, we define the PLUMED input as a multi-line string in
Python. We consider the particular case of applying a metadynamics bias to
a collective variable, in this case the unit cell volume.
Because metadynamics represents a time-dependent bias,
it relies on an additional _hills_ file which keeps track of the location of
Gaussian hills that were installed in the system at various steps throughout
the simulation. Psiflow automatically takes care of such external files, and
their file path in the input string is essentially irrelevant.
```py
from psiflow.sampling import DynamicWalker, PlumedBias


plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=10 LABEL=metad FILE=dummy
"""
bias = PlumedBias(context, plumed_input)        # a new hills file is generated

walker = DynamicWalker()
state = walker.propagate(model, bias=bias)      # this state is obtained through biased MD

```
Note that the bias instance will retain the hills it generated during walker
propagation.
Let's say we wanted to investigate what our training data from before looks like
in terms of collective variable distribution and bias energy.
To facilitate this, psiflow provides the ability to evaluate `PlumedBias` objects
on `Dataset` instances using the `bias.evaluate()` method.

```py
values = bias.evaluate(data_train, variable='CV')       # evaluate all PLUMED actions with ARG=CV on data_train

assert values.result().shape[0] == data_train.length().result()  # each snapshot is evaluated separately
assert values.result().shape[1] == 2                             # CV and bias per snapshot, in PLUMED units!
assert not np.allclose(values.result()[:, 1], 0)                 # nonzero bias energy
```
As another example, let's consider the same collective variable but now with
a harmonic bias potential applied to it.
Because sampling with and manipulation of harmonic bias potentials is ubiquitous
in free energy calculations, psiflow provides specific support for this.
```py
plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
RESTRAINT ARG=CV AT=150 KAPPA=1 LABEL=restraint
"""
bias  = PlumedBias(context, plumed_input)
state0 = walker.propagate(model, bias=bias)                 # propagation with bias centered at CV=150

bias.adjust_restraint(variable='CV', kappa=2, center=200)   # decrease width and shift center to higher volume
state1 = walker.propagate(model, bias=bias)     

# if the system had enough time to equilibrate with the bias, then the following should hold
assert state0.result().get_volume() < state1.result().get_volume()

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
bias = PlumedBias(context, plumed_input, data={'EXTERNAL': grid})   # pass grid file as external dependency
```
Bias potentials can also be added to an `Ensemble`, either upon initialization
or using the `ensemble.add_bias()` method.
Note that metadynamics hills files cannot be shared between walkers 
(as is the case in multiple walker metadynamics) as this would
violate their strict independence.

!!! note 
    PLUMED interfacing is not supported for the `OptimizationWalker` because
    (i) it is rarely ever useful to add a bias during optimization, and (ii)
    ASE's PLUMED interface is shaky at best.


## Level of theory
Data sampled using the `Ensemble` should be labeled with the correct QM energy,
force, and virial stress before it can be used during model training.
The `BaseReference` class implements the singlepoint evaluations using specific
QM software packages and levels of theory.
At the moment, psiflow only supports CP2K as the reference level of theory,
though VASP and ORCA will be added in the near future.
### CP2K
The `CP2KReference` expects a traditional CP2K
[input file](https://github.com/svandenhaute/psiflow/blob/main/examples/data/cp2k_input.txt)
(again represented as a multi-line string in Python, just like the PLUMED input);
it should only contain the FORCE_EVAL section.
Additional input files which define the basis sets, pseudopotentials, and
dispersion correction parameters can be added to the calculator after initialization.
```py
from psiflow.reference import CP2KReference


cp2k_input = with file('cp2k_input.txt', 'r') as f: f.read()
reference  = CP2KReference(context, cp2k_input)

# register additional input files with the following mapping
# if the corresponding keyword in the CP2K input file is X, use Y as key here:
# X: BASIS_SET_FILE_NAME    ->   Y: basis_set
# X: POTENTIAL_FILE_NAME    ->   Y: potential
# X: PARAMETER_FILE_NAME    ->   Y: dftd3
reference.add_file('basis_set', 'BASIS_MOLOPT_UZH')
reference.add_file('potential', 'POTENTIAL_UZH')
reference.add_file('dftd3', 'dftd3.dat')

unlabeled = ensemble.sample(model=model)            # sample some states using some ensemble
labeled = reference.evaluate(unlabeled)             # perform parallel singlepoint of all snapshots

print(labeled[0].result().info['energy'])           # cp2k potential energy!
```


## Learning Algorithms
The following is a (simplified) excerpt that illustrates how these basic
building blocks may be used to implement a simple online
learning approach:

```py
# parameters (dataclass)        : defines number of iterations and the number of states to sample
# model (type BaseModel)        : represents a trainable potential (e.g. NequIP)
# data_train (type Dataset)     : represents the initial training and validation data
# data_valid
# ensemble (type Ensemble)      : defines phase space sampling (e.g. 20 parallel MD runs)
# reference (type BaseReference): defines the QM level of theory and calculator (e.g. CP2K, PBE-D3(BJ)+TZVP)

for i in range(parameters.niterations):
    model.deploy() # performs e.g. torch.jit.compile in desired precision

    # ensemble wraps a set of phase space walkers (e.g. multiple NPT simulations)
    dataset = ensemble.sample(
            parameters.nstates, # sample this number of states
            model=model, # use current best model as potential energy surface
            )
    data = reference.evaluate(dataset) # massively parallel QM evaluation of sampled states
    data_success = data.get(indices=data.success) # some calculations may fail!
    train, valid = get_train_valid_indices(
            data_success.length(),
            self.parameters.train_valid_split,
            )
    data_train.append(data_success.get(indices=train))
    data_valid.append(data_success.get(indices=valid))

    if parameters.retrain_model_per_iteration: # recalibrate scale/shift/avg_num_neighbors
        model.reset()
        model.initialize(data_train)

    epochs = model.train(data_train, data_valid) # train model for some time
```
[TODO]

