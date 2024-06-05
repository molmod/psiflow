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
It is initialized with a `Geometry` instance which describes the start of the simulation, and can be assigned a particular hamiltonian, a temperature and/or pressure, and a timestep.

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
In the vast majority of cases, it is necessary to run mutiple simulations at slightly different conditions.
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
Importantly, there are *no restriction* on the type of walkers in this list.
Users can mix regular NVT walkers, with PIMD NVE walkers, and a list of N replica exchange walkers.
Internally, psiflow will recognize which walkers are independent and parallelize the execution as much as possible
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
In this example, the sample function will return a list of `Output` instances, each of which contains the trajectory of a single walker.
The outputs are ordered in the same way as the input walkers (i.e. `outputs[0]` corresponds to the output from `walkers[0]`).
They provide access to the sampled trajectory of the simulation, the elapsed simulation time, and importantly, a number of *observable properties*
which have been written out by i-PI. These properties can be used to compute averages of physical observables, such as the internal energy or the virial stress tensor.
A full list of available properties is given in the [i-PI documentation](https://ipi-code.org/i-pi/output-tags.html). Note that psiflow adheres to the same naming convention as adopted in i-PI:

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

As a sanity check, let us recompute the potential energies which were stored at each snapshot during the simulation using the `evaluate` functionality of our MACE hamiltonian:
```py
mace   = walkers[0].hamiltonian                               # the hamiltonian used in the simulations
future = mace.evaluate(outputs[0].trajectory).get('energy')   # future of the recomputed energies as an array

manual_energies_0 = future.result()                           # get the actual numpy array

assert np.allclose(
    manual_energies_0,
    energies[0],
    )
```
## walker utilities

## PIMD simulations

## replica exchange

## metadynamics
