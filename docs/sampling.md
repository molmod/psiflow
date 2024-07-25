In the Born-Oppenheimer philosophy, we explore the phase space of a molecule or a material and generate samples using molecular dynamics simulations.
Those samples are then used to evaluate time averages of some property of interest in order to predict physical observables.
In psiflow, such simulations are executed within [i-PI](https://ipi-code.org/), a versatile and efficient code which supports an impressive number of [features](https://ipi-code.org/i-pi/features.html).
We mention the most important ones below

- **molecular dynamics in various ensembles**: most notably NVE, NVT, and fully anisotropic NPT. There exist a variety of thermostat and barostat options, the default being Langevin. Together with the ability to combine arbitrary hamiltonians, this includes biased molecular dynamics simulations using e.g. harmonic restraints (umbrella sampling).
- **path-integral molecular dynamics** (PIMD): allows for the simulation of the quantum behavior of light atomic nuclei. This is important for many systems involving hydrogen atoms at relatively low temperatures (<=room temperature). Importantly these simulations can also be performed in any of the aforementioned ensembles. 
- **geometry optimizations**: i-PI can be used to optimize the geometry of a molecule or a material using a variety of optimization algorithms.
- **replica exchange** (parallel tempering): dramatically improves the sampling efficiency and ergodicity whenever nontrivial free energy barriers are present in the phase space of the system. In this approach, one considers replicas of the system at various temperatures and/or pressures, or optionally even with different hamiltonians.
- **multiple walker metadynamics**: simple but powerful method to overcome free energy barriers when a suitable collective variable is known for the system of interest.


## the `Walker` class
Psiflow is essentially a convenient wrapper around most of i-PI's features.
The key object which enables the execution of these simulations is the `Walker` class.
A single walker describes a single replica of the system which evolves through phase space.
It is initialized with a `Geometry` instance which describes the start of the simulation.
Optional arguments include a hamiltonian for the walker (which is used to evaluate forces
during simulations), the temperature and pressure, or a timestep.

```py
from psiflow.sampling import Walker
from psiflow.geometry import Geometry
from psiflow.hamiltonians import get_mace_mp0


start = Geometry.load("start.xyz")
walker = Walker(
    start,
    hamiltonian=get_mace_mp0(),
    temperature=300.0,
    pressure=None,  # NVT simulation
    timestep=0.5,   # in femtoseconds, the default value
)
```
In the vast majority of cases, it is necessary to run multiple simulations at slightly different conditions.
For example, let us create ten of these walkers which are identical except for the temperature:

```py
walkers = walker.multiply(10)
for i, walker in enumerate(walkers):
    w.temperature = 300 + i * 10
```
When propagated, each of these walkers will generate trajectories in phase space which correspond to their own temperature.
In the case of temperature, such trajectories can be used to e.g. evaluate variation of the mean energy with respect to temperature
(and therefore, the heat capacity of the system).

## generating trajectories

Walkers can be propagated in time by using the `sample` function.
It accepts a list of walkers, each of which will be propagated in phase space according to its own parameters.
Importantly, there are *no restrictions* on the type of walkers in this list.
Users can mix regular NVE walkers, with path-integral NPT walkers, and a list of N replica exchange walkers.
Internally, psiflow will recognize which walkers are independent and parallelize the execution as much as possible.
Consider the following example:
```py
from psiflow.sampling import sample

outputs = sample(
    walkers,
    steps=1e6,  # total number of timesteps to run the simulation for; this translates to 500 ps in this case
    step=1e3,   # sample every 1000 timesteps
    start=1e5,  # start sampling after 50 ps
)
print(outputs)  # list of `SimulationOutput` instances
```
In this example, the sample function will return a list of `Output` instances, each of which contains the trajectory of a single walker,
as obtained by performing molecular dynamics simulations in the corresponding ensemble and with the provided total number of steps.
Note that each of these simulations is independent (there exists no coupling between them), and as such, they will be executed in parallel as much as possible.
The outputs are ordered in the same way as the input walkers (i.e. `outputs[0]` corresponds to the output from `walkers[0]`).
They provide access to the sampled trajectory of the simulation, the elapsed simulation time, and importantly, a number of *observable properties*
which have been written out by i-PI. These properties can be used to compute averages of physical observables, such as the internal energy or the virial stress tensor.
A full list of available properties is given in the [i-PI documentation](https://ipi-code.org/i-pi/output-tags.html)
(note that psiflow adheres to the same naming convention as adopted in i-PI).
The following options are kept track of by default:

- `energy`: the total energy of the system. The actual name of this quantity is `potential{electronvolt}`
- `temperature`: the instantaneous temperature of the system. The actual name of this quantity is `temperature{kelvin}`
- `time`: the elapsed simulation time. The actual name of this quantity is `time{picosecond}`
- `volume`: the volume of the simulation cell (only for periodic systems). The actual name of this quantity is `volume{angstrom3}`

Similarly to the evaluation of a `Hamiltonian` or the querying of a snapshot in a `Dataset`, simulation outputs are returned as futures.
For example, say we wanted to compute the average energy for each of the simulations:
```py
import numpy as np

energy_futures = [output["potential{electronvolt}"] for output in outputs]
energies = [future.result() for future in energy_futures]
mean_energy = np.array([np.mean(energy) for energy in energies])
```
This example extracts the futures which contain the potential energies of all simulations, waits for them to complete (via `result()`), and then computes the mean energy for each simulation. In a very similar fashion, we can compute the bulk modules of bcc iron simply by constructing walkers at various pressures and extracting the corresponding `volume{angstrom3}` observable -- see [here](https://github.com/molmod/psiflow/tree/main/examples/iron_bulk_modulus.py).

In many cases, it is useful to save the trajectory of a simulation to disk.
Trajectories are essentially just a series of snapshots, and as such, psiflow represents them as `Dataset` instances.
Each of the outputs has an attribute `trajectory` which is a `Dataset` instance.
Let us save the trajectory of the first simulation to disk:

```py
outputs[0].trajectory.save("300K.xyz")
```

As a sanity check, let us recompute the potential energies which were stored at each snapshot during the simulation using the `compute` functionality of our MACE hamiltonian:
```py
mace   = walkers[0].hamiltonian                               # the hamiltonian used in the simulations
future = mace.compute(outputs[0].trajectory, 'energy')        # future of the recomputed energies as an array

manual_energies_0 = future.result()                           # get the actual numpy array

assert np.allclose(
    manual_energies_0,
    energies[0],
    )
```
Trajectories are only saved by default if a `step` argument is provided, which determines
the saving frequency for both the output properties (such as energy or temperature) as
well as for the trajectory.
In some cases, one wishes to save output properties without actually saving the
trajectory because it would end up consuming too much disk space.
For this use case, it is possible to pass `keep_trajectory=False` as an additional
argument to the sample function.
A full list of possible arguments is given below:

- **walkers** (type `list[Walker]`): the walkers which should be propagated. Internally,
  they are partitioned into groups such that e.g. replica-exchange-coupled walkers are
  propagated at the same time (and on the same node) in order to allow for communication
  between them.
- **steps** (type `int`): the total number of steps to perform in each of the simulations.
- **step** (type `int`): saving frequency of both output properties as well as the
  trajectory.
- **start** (type `int`): properties and snapshots will not be saved before `start` steps
  have passed -- used for equilibration.
- **keep_trajectory** (type `bool`): determines whether or not to save the trajectory.
- **max_force** (type `float`, in eV/A): determines the maximum value of any of the forces
  in the walker before a simulation is interrupted. Large forces are typically a result of
  an unphysical region in phase space and/or a badly-trained ML potential.
- **observables** (type `list[str]`): physical properties which are to be tracked and
  saved during the simulation. An exhaustive list of all options is given in the
  [i-PI documentation](https://ipi-code.org/i-pi/output-tags.html). Four quantities are
  saved by default: energy, temperature, time, and unit cell volume.
- **fix_com** (type `bool`): whether or not to fix the center of mass. Usually a good idea
  -- default is `True`.
- **prng_seed** (type `int`): default RNG seed for i-PI, in order to be able to reproduce
- **use_unique_seeds** (type `bool`): use a unique seed for each of the walkers, starting
  at the given `prng_seed` for walker 0, `prng_seed + 1` for walker 1, etc.
- **checkpoint_step** (type `int`): how frequently to save checkpoints. Checkpoints are
  used at the end of a simulation in order to update the `state` attribute of the walker.
  If the simulation is (gracefully) terminated because of e.g. a walltime limit on the
  SLURM node, then the state of the walker will be determined by the last saved
  checkpoint.
- **verbosity** (type `str`): verbosity of the i-PI output files; `low`, `medium`, or
  `high`.
- motion_defaults (type `ET.Element`): *TODO*: additional thermostat and barostat options

## randomize and quench

If molecular simulation would rigorously satisfy the ergodic hypothesis, then the starting structure of a simulation is entirely irrelevant and will not affect the sampling distribution.
In practice, however, the ergodic hypothesis is almost always violated due to a number of reasons, and the starting structure can have a significant impact on the subsequent simulation.

If the starting structure is too strongly out of equilibrium for the specific hamiltonian of the walker (i.e. the initial energy and forces are too large), the simulation will likely explode in the first few steps and it will practically never return physically relevant samples.
Because each walker can have its own hamiltonian, a given geometry might be an entirely infeasible start for one walker, but be reasonable for another.
Psiflow provides a simple and efficient method to *assign* reasonable starting structures to a collection of walkers based on a large set of possible candidates: the `quench` method.
It accepts a list of `Walker` instances, and a `Dataset` of possible candidate structures.
It will automatically evaluate the hamiltonians of the walkers over the dataset in an efficient manner, and assign the structure with the lowest energy to each walker.
By applying the `quench` method, psiflow reassigns walker starting geometries to the lowest energy structures as obtained from a dataset of choice:
```py
from psiflow.sampling import quench

# assume 'walkers' is a list of `Walker` instances with badly initialized geometries

data = Dataset.load("my_large_dataset_of_candidates.xyz")
quench(walkers, data)

relaxed = Dataset([w.hamiltonian.evaluate(w.start) for w in walkers])

energies = relaxed.get('energy').result()   # energies are now much lower
```

In an alternative setting, it might be useful to randomize the starting structure for each walkers from a given dataset (e.g. to improve sampling coverage). This can be achieved using the `randomize` method:
```py
from psiflow.sampling import randomize

randomize(walkers, data)
```
Randomizing walker initializations is particularly useful during active learning and in
case different walkers are sampling different ensembles (e.g. different temperatures,
pressures, bias potentials).

## advanced options
The real power of i-PI lies in its extensive support for advanced sampling algorithms.
This section discusses the most important ones which are currently supported within psiflow.

### PIMD simulations
The Born-Oppenheimer approximation breaks down for light nuclei (most notably hydrogen),
and by adapting traditional molecular dynamics algorithms, it is possible to account
for the quantum mechanical (wave-like) nature of protons during simulations.
These techniques are referred to as path-integral methods and represent the core business
of i-PI.
In practice, it comes down to running N replicas of the same simulation in parallel, in
which consecutive replicas are coupled by harmonic spring forces. As such, the cost of
these simulations is N times higher than a classical simulation of the same system, and in
many cases also the time step needs to be chosen smaller (e.g. 0.25 fs instead of 0.5 fs).

Performing PIMD simulations in psiflow is pretty much identical to 'normal' simulations;
walkers can be given an `nbeads` attribute which determines the number of replicas. Its
default value is 1, meaning purely classical MD.

```py
pimd_walker = Walker(
    geometry,
    hamiltonian,
    temperature=200,
    pressure=0.1,       # PIMD is compatible with NVE, NVT, and even NPT
    nbeads=32,          # run 32 replicas in parallel
    timestep=0.4,       # in fs
)

classical_walker = Walker(
    geometry,
    hamiltonian,
    temperature=600,
    pressure=None,
    nbeads=1,           # the default value
    timestep=1,         # in fs
)

# simulate two walkers:
# the first will propagate 32 individual replicas, the second only one
outputs = sample(
    [pimd_walker, classical_walker],
    steps=10000,
)

```
Because each of the beads is equivalent, only one of the beads is used to save an output
trajectory. Similarly, the `start` and `state` attributes which track the current and
start position of the walker only keep track of one replica.


### replica exchange
Replica exchange (otherwise called parallel tempering) is a technique used to improve the
ergodicity of sampling algorithms.
It consists of simulating multiple replicas of the same system in parallel but at
different thermodynamic conditions.
This implies different temperatures, different pressures, or even different hamiltonians.
During the simulation, Monte Carlo trial moves are attempted which try to swap atomic geometries
between to replicas. If trial moves are rejected according to the specific acceptance
criterion, it can be shown that replica exchange does not alter the theoretical sampling
distribution of each of the replicas. The only thing it does is make the sampling less
dependent on the initial starting configuration.

Replica exchange can be enabled between walkers of the same ensemble (all NVE, all NVT, or
all NPT) simply by calling the `replica_exchange` function. The i-PI implementation
which psiflow uses requests two additional parameters; see the
[documentation](https://ipi-code.org/i-pi/input-tags.html#remd) and the
[associated paper](https://onlinelibrary.wiley.com/doi/10.1002/jcc.24025) for more details.

```py
from psiflow.sampling import replica_exchange


walkers = []
for temperature, pressure in zip([300, 350, 400], [-10, 0, 10]):
    walker = Walker(
        start=geometry,
        hamiltonian=hamiltonian,
        temperature=temperature,
        pressure=pressure,
    )
    walkers.append(walker)

# add replica exchange to walkers; modifies them in-place
replica_exchange(
    walkers,
    trial_frequency=100,        # trial frequency, see i-PI docs
    rescale_kinetic=True,       # usually a good idea, see i-PI docs
)

output = sample(walkers, steps=10000, step=100)
```

!!! note "coupled simulations run single-node"
    PIMD and replica exchange simulations involve N parallel replicas of the same system
    which are _dependent_ on each other for their time propagation. As such, they are
    scheduled by psiflow in a way that guarantees that they run simultaneously and that
    they can communicate with each other without too much hassle. At the moment, this
    implies that all replicas need to be executed on a single node. If you have powerful
    nodes with four or eight GPUs and relatively lightweight MACE models, this is usually not a problem.
    For more large-scale applications, some manual hacking will be necessary. Open an
    issue on Github if you need help on this.

### metadynamics
Metadynamics (and its variants) are an extremely common form of enhanced sampling and
phase space exploration, and i-PI has native support for these using PLUMED.
Similarly to how time-independent bias contributions are created simply based on their
input string, psiflow implements metadynamics using a `Metadynamics` object which is
constructed based on a plumed input file.

For the proton jump example in vinyl alcohol, an equivalent metadynamics input would look
like this:

```py
from psiflow.sampling import Metadynamics


plumed_str = """UNITS LENGTH=A ENERGY=kj/mol
d_C: DISTANCE ATOMS=3,5
d_O: DISTANCE ATOMS=1,5
CV: COMBINE ARG=d_C,d_O COEFFICIENTS=1,-1 PERIODIC=NO
METAD ARG=CV PACE=10 SIGMA=0.1 HEIGHT=5

"""

metadynamics = Metadynamics(plumed_str)

walker = Walker(geometry, hamiltonian, temperature=300, metadynamics=metadynamics)
outputs = sample([walker], steps=10000, step=100)

# added hills during simulation are tracked as a parsl DataFuture
hills_file = metadynamics.external  # 10,000 / 10 hills

```

The corresponding hills file which keeps track of the trajectory of the simulation in CV
space (as well as a list of the Gaussian hills) is stored in the `metadynamics.external`
attribute. Since this is, like any observable or trajectory, the outcome of a simulation,
psiflow uses futures to represent it. In this case, the hills file is therefore a future
-- a `DataFuture` to be precise. It can be read and analyzed much like a regular hills
file, except that we need to make sure to call `.result()` in order to wait until the
simulation has completed.

```py

metadynamics.external.result()  # call result() to wait until simulation has finished

# read it
with open(metadynamics.external.filepath, 'r') as f:
    content = f.read()

print(content)

```

## geometry optimizations

Aside from molecular dynamics, it is often very useful to use gradient-based optimization
on the potential energy surface to 'descend' into (most likely) a local energy
minimum.
While this could theoretically be considered as a kind of zero-kelvin walker, geometry
optimization is implemented in a separate module, and does not require any walkers.
Instead, psiflow exposes a single `optimize` function which requires the following
arguments:

- **state** (type `Geometry`): the initial state of the system.
- **hamiltonian** (type `Hamiltonian`): the hamiltonian defines the PES to be optimized
- **steps** (type `int`): the maximum number of optimization steps.
- **keep_trajectory** (type `bool`): whether to keep the optimization trajectory
- **mode** (type `str`): optimization algorithm, see [i-PI
  documentation](https://ipi-code.org/i-pi/input-tags.html#optimizer) for details.
- **etol** (type `float`): convergence tolerance on the energy (eV)
- **ptol** (type `float`): convergence tolerance on the positions (angstrom)
- **ftol** (type `float`): convergence tolerance on the forces (eV/angstrom)
