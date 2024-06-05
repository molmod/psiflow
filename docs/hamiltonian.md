In Born-Oppenheimer-based molecular simulation, atomic nuclei are treated as classical particles that are subject to *effective* interactions, which are determined by the quantum mechanical behavior of the electrons.
These interactions determine the interatomic forces which are used in a dynamic simulation to propagate the atomic positions from one timestep to the next.
In more advanced schemes, researchers may modify these effective interactions to include biasing forces (e.g. in order to induce a phase transition), or perform an alchemical transformation between two potential energy surfaces (when computing relative free energies).

The ability to combine various energy contributions in an arbitrary manner allows for a very natural definition of many algorithms in computational statistical physics.
To accomodate for all these use cases, psiflow provides a simple abstraction for *"a function which accepts an atomic geometry and returns energies and forces"*: the `Hamiltonian` class.
Examples of Hamiltonians are a specific ML potential, a bias potential on a collective variable, or a quadratic approximation to a potential energy minimum.

By far the simplest hamiltonian is the Einstein crystal, which binds atoms to a certain reference position using harmonic springs with a single, fixed force constant.

```py
from psiflow.geometry import Geometry
from psiflow.hamiltonians import EinsteinCrystal


geometry = Geometry.from_string('''
    2
    H 0.0 0.0 0.0
    H 0.0 0.0 0.8
''')

einstein = EinsteinCrystal(
    reference_geometry=geometry.positions,  # positions at which all springs are at rest
    force_constant=0.1,                     # force constant, in eV / A**2
    )

```
As mentioned earlier, the key feature of hamiltonians is that they take as input an atomic geometry, and spit out an energy, a set of forces, and optionally also virial stress.
Because hamiltonians might require specialized resources for their evaluation (e.g. an ML potential which gets executed on a GPU), evaluation of a hamiltonian does not necessarily happen instantly (e.g. if a GPU node is not immediately available). Similar to how `Dataset` instances return futures of a `Geometry` when a particular index is queried, hamiltonians return a future when asked to evaluate the energy/forces/stress of a particular `Geometry`:

```py
future = einstein.evaluate(geometry)      # returns an AppFuture of the Geometry; evaluates instantly
evaluated = future.result()                  # calling result makes us wait for it to actually complete

assert evaluated.energy is not None                     # the energy of the hamiltonian
assert not np.any(np.isnan(evaluated.per_atom.forces))  # a (N, 3) array with forces
```
One of the most commonly used hamiltonians will be that of MACE, one of the most ubiquitous ML potentials.
There exist reasonably accurate pretrained models which can be used for exploratory purposes. 
These are readily available in psiflow:

```py
from psiflow.hamiltonians import get_mace_mp0


mace = get_mace_mp0()               # downloads MACE-MP0 from github
future = mace.evaluate(geometry)    # evaluates the MACE potential on the geometry

evaluated = future.result()
forces = evaluated.per_atom.forces  # forces on each atom, in float32

assert np.sum(np.dot(forces[0], forces[1])) < 0  # forces in H2 always point opposite of each other
assert np.allclose(np.sum(forces), 0.0)          # forces are conservative --> sum to zero
```
As alluded to earlier, hamiltonians can be combined in arbitrary ways to create new hamiltonians.
Psiflow supports a concise syntax for basic arithmetic operations on hamiltonians, such as 
multiplication by a scalar or addition of two hamiltonians:

```py
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
