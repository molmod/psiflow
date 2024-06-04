In Born-Oppenheimer-based molecular simulation, atomic nuclei are treated as classical particles that are subject to *effective* interactions which are determined by the quantum mechanical behavior of the electrons.
In addition to the atomic interactions, it is often useful to define additional biasing forces on the system, e.g. in order to drive a rare event or to prevent the system from exploring undesired regions in phase space.
In addition, there exist various alchemical free energy techniques which rely on systematic changes in the hamiltonian ( = potential energy) of the system to derive free energy differences between different states.

To accomodate for all these use cases, psiflow provides a simple abstraction for *a function which accepts an atomic geometry and returns energies and forces*: the `Hamiltonian` class.
The simplest hamiltonian (which is only really useful for testing purposes) is the Einstein crystal, which binds atoms using harmonic springs to a certain reference position.
```py

geometry = Geometry.from_string('''
    2
    H 0.0 0.0 0.0
    H 0.0 0.0 0.8
''')

hamiltonian = EinsteinCrystal(
    reference_geometry=geometry.positions,
    force_constant=0.1,
    )

```
