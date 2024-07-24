In Born-Oppenheimer-based molecular simulation, atomic nuclei are treated as classical
particles that are subject to *effective* interactions -- these are the result of the quantum
mechanical behavior of the electrons. These interactions determine the interatomic forces
which are used in a dynamic simulation to propagate the atomic positions from one timestep
to the next.
Traditionally, dynamic simulations required an explicit evaluation of these effective
forces in terms of a quantum mechanical calculation (e.g. DFT(B)).
Recently, it became clear that it is much more efficient to perform such simulations
using a machine-learned representation of the interaction energy, i.e. an ML potential. 
The development and application of ML potentials throughout large simulation workflows is in
fact one of the core applications of psiflow.

The `Hamiltonian` class is used to represent any type of interaction potential.
Examples are pre-trained, 'universal' models (e.g. [MACE-MP0](https://arxiv.org/abs/2401.00096)),
ML potentials trained within psiflow (see [ML potentials](model.md)), or a quadratic
(hessian-based) approximation to a local energy minimum, to name a few.
In addition, various sampling schemes employ bias potentials which are superimposed on the
QM-based Born-Oppenheimer surface in order to drive the system
along specific reaction coordinates (e.g. metadynamics, umbrella sampling).
Such bias potentials are also instances of a `Hamiltonian`.

By far the simplest hamiltonian is the Einstein crystal, which binds atoms to a certain
reference position using harmonic springs with a single, fixed force constant.

```py
from psiflow.geometry import Geometry
from psiflow.hamiltonians import EinsteinCrystal


# isolated H2 molecule
geometry = Geometry.from_string('''
    2
    H 0.0 0.0 0.0
    H 0.0 0.0 0.8
''')

einstein = EinsteinCrystal(geometry, force_constant=0.1)  # in eV/A**2

```
As mentioned earlier, the key feature of hamiltonians is that they represent an interaction energy between atoms,
i.e. they output an energy (and its gradients) when given a geometry as input.
Because hamiltonians might require specialized resources for their evaluation (e.g. an ML
potential which gets executed on a GPU), evaluation of a hamiltonian does not necessarily
happen instantly (e.g. if a GPU node is not immediately available). Similar to how
`Dataset` instances return futures of a `Geometry` when a particular index is queried,
hamiltonians return a future when asked to evaluate the energy/forces/stress of a
particular `Geometry`:

```py
energy = einstein.compute(geometry, 'energy')       # AppFuture of an energy (np.ndarray with shape (1,))
print(energy.result())                              # wait for the result to complete, and print it (in eV)


data = Dataset.load('snapshots.xyz')                # N snapshots
energy, forces, stress = einstein.compute(data)     # returns energy and gradients for each snapshot in data


assert energy.result().shape == (N,)                # one energy per snapshot
assert forces.result().shape == (N, max_natoms, 3)  # forces for each snapshot, with padded natoms
assert stress.result().shape == (N, 3, 3)           # stress; filled with NaNs if not applicable
```
An particularly important hamiltonian is MACE, one of the most ubiquitous ML potentials.
These are readily available in psiflow:

```py
from psiflow.hamiltonians import MACEHamiltonian


mace = MACEHamiltonian.mace_mp0()                   # downloads MACE-MP0 from github
forces = mace.compute(geometry, 'forces')           # evaluates the MACE potential on the geometry

forces = forces.result()                            # wait for evaluation to complete and get actual value

assert np.sum(np.dot(forces[0], forces[1])) < 0     # forces in H2 always point opposite of each other

assert np.allclose(np.sum(forces, axis=0), 0.0)     # forces are conservative --> sum to [0, 0, 0]
```
A unique feature of psiflow `Hamiltonian` instances is the ability to create a new
hamiltonian from a linear combination of two or more existing hamiltonians.
Let us consider the particular example of [umbrella
sampling](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.66) using
[PLUMED](https://www.plumed.org/):

```py
from psiflow.hamiltonians import PlumedHamiltonian

plumed_str = ""
bias = PlumedHamiltonian()

```

$$
H = \alpha H_0 + (1 - \alpha) H_1
$$

is very straightforward to express in code:


This allows for a very natural definition of many algorithms in computational statistical physics
(e.g. Hamiltonian replica exchange, thermodynamic integration, biased dynamics).

```py
from psiflow.hamiltonians import PlumedHamiltonian


plumed_input = """

"""
bias = 


data = Dataset.load('train.xyz')
mix = 0.5 * einstein + 0.5 * mace             # MixtureHamiltonian with E = 0.5 * E_einstein + 0.5 * E_mace
energies_mix = mix.evaluate(data).get('energy')

energies_einstein = einstein.evaluate(data).get('energy')
energies_mace     = mace.evaluate(data).get('energy')
assert np.allclose(
      energies_mix.result(),
      0.5 * energies_einstein.result() + 0.5 * energies_mace.result(),
      )
```
This makes it very easy to introduce bias potentials into your simulations -- see for example the formic acid transition state [example](https://github.com/molmod/psiflow/tree/main/examples/formic_acid_transition.py).
The following is a list of all available hamiltonians in psiflow:

- `EinsteinCrystal`: A simple harmonic potential which binds atoms to a reference position.
- `MACE`: ML potential, either pretrained as available on GitHub, or trained within psiflow (see later sections)
- `Harmonic`: A general quadratic potential based on a Hessian matrix and an optimized geometry.
- `PlumedHamiltonian`: a bias contribution based on a PLUMED input file.
