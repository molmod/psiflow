## representing a single atomic structure

A key component of any molecular simulation engine is the ability to represent an ordered collection of atoms in 3D space.
In psiflow, this is achieved using the `Geometry` class, which is essentially a lite
version of ASE's `Atoms` object.
A `Geometry` instance describes a set of atoms, each of which is characterized by its position in space, its chemical identity, and, if available, a force which acts on the atoms.
In addition, atomic geometries can also contain metadata such as information on periodicity in 3D space (unit cell vectors), the potential energy, a stress tensor, or the value of a particular order parameter.

Atomic geometries can be created from an XYZ string, from an existing ASE atoms instance, or directly from raw arrays of positions, atomic numbers, and (optionally) unit cell vectors.
```py
import ase
import numpy as np
from psiflow.geometry import Geometry


# a simple H2 molecule in vacuum
geometry = Geometry.from_string('''
    2
    H 0.0 0.0 0.0
    H 0.0 0.0 0.8
''')

# the same H2 molecule using ase Atoms
atoms = ase.Atoms(
            numbers=[1, 1, 1],
            positions=[[0, 0, 0], [0, 0, 0.8]],
            pbc=False,
            )
geometry = Geometry.from_atoms(atoms)  # creates an identical instance

print(len(geometry))                   # prints the number of atoms, in this case 2
assert geometry.pbc == False           # if no cell info is given, the instance is assumed to be non-periodic
geometry.cell[:] = 10 * np.eye(3)      # set the cell vectors to a 10 A x 10 A x 10 A cube
assert geometry.pbc == True            # now the instance is periodic


print(geometry.energy)                    # None; no energy has been set
print(np.all(np.isnan(geometry.forces)))  # True; no forces have been set


# the same instance, directly from numpy
geometry = Geometry.from_data(
          positions=np.array([[0, 0, 0], [0, 0, 0.8]]),
          numbers=np.array([1, 1]),
          cell=None,
          )

```

All features in psiflow are fully compatible with either nonperiodic (molecular) or 3D periodic systems with arbitrary unit cells (i.e. general triclinic).
For periodic systems in particular, it is common to require that atomic geometries are represented in their *canonical* orientation.
In this (unique) orientation, cell vectors are aligned with the X, Y, and Z axes as much as possible, with the convention that the X-axis is aligned with the first cell vector, the Y-axis is oriented such that the second cell vector lies in the XY-plane.
In addition, box vectors are added and subtracted in order to make the cell as orthorhombic as possible.
When the cell vectors are represented as the rows of a 3-by-3 matrix, then the canonical orientation ensures
that this matrix is _lower-triangular_.
Note that this transformation changes nothing about the physical behavior of the system
whatsoever; interatomic distances or unit cell volume remain exactly the same.

```py
geometry = Geometry.from_data(
          positions=np.array([[0, 0, 0], [0, 0, 0.8]]),
          numbers=np.array([1, 1]),
          cell=np.array([[4, 0, 0], [0, 4, 0], [3, 3, 6]]),
          )
geometry.align_axes()           # transform into canonical representation
print(geometry.cell)            # third vector: [3, 3, 6] --> [-1, -1, 6]
```
The canonical orientation is convenient for a number of reasons, but the main one is
computational efficiency: application of the minimum image convention and/or neighbor list
construction becomes *much* more elegant. For this reason, a number of molecular
simulation engines (i-PI, GROMACS, OpenMM, to name a few) require that the starting
configuration of the system is given in its canonical orientation.


To understand what is meant by 'generating data in the future', consider the following
example: imagine that we have a trajectory of atomic geometries, and we wish to
minimize each of their potential energies and inspect the final optimized geometry for
each state in the trajectory. In pseudo-code, this would look something like this:

```
for state in trajectory:
    final = geometry_optimization(state)
    detect_minimum(final)

```
In "normal", _synchronous_ execution, when entering the first iteration of the loop, Python would
start executing the first geometry optimization right away and *wait* for it complete, before
moving on to the next iteration. This makes little sense, since we know in advance
that each of the optimizations is in fact independent. As such, the loop can in fact be completed
much quicker if we would simply execute each optimization in parallel (as opposed to
serial).
The intended way to achieve this in Python is by using the built-in
[_concurrent_](https://docs.python.org/3/library/concurrent.futures.html) library,
and it provides the foundation of psiflow's efficiency and scalability.
Aside from _asynchronous_ execution, we also want _remote_ execution.
Geometry optimizations, molecular dynamics simulations, ML training, quantum chemistry
calculations, ... are rarely ever performed on a local laptop.
Instead, they should ideally be forwarded towards e.g. a SLURM/PBS(Pro) scheduler or an
AWS/GCP instance.
To achieve this, psiflow is built with
[Parsl](https://github.com/parsl/parsl): a DoE-funded Python package which
extends the native _concurrent_ library with
the ability to offload execution towards remote compute resources.

Parsl (and `python.concurrent`) is founded on two concepts: apps and futures. In their simplest
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
For more information, check out the [Parsl documentation](https://parsl.readthedocs.io/en/stable/).

In our geometry optimization example from before, we would implement the function
`geometry_optimization` as a Parsl app, and its return value `final` would no longer
represent the actual optimized geometry; it would be a future of the optimized geometry.
Importantly, when organized in this way, Python will go through the loop almost
instantaneously, observe that we want to perform a number of `geometry_optimization`
calculations, and offload those calculations separately, independently, and immediately to whatever compute resource
it has available. As such, all optimizations will effectively run in parallel:
```
for state in trajectory:
    final_future = geometry_optimization_app(state)  # completes instantaneously!
    detect_minimum_app(final_future)                 # completes instantaneously!
    type(final_future)                               # AppFuture

# all geometry optimizations are running

print(final_future.result())    # print the obtained `Geometry` from the last loop

```


## representing multiple structures

In many cases, it is necessary to represent a collection of atomic configurations, for example, a trajectory of snapshots generated by molecular dynamics, or a dataset of atomic configurations used for model training.
In psiflow, such collections are represented using the `Dataset` class.

Naively, the most straightforward representation of multiple atomic structures would
simply be a `list` of `Geometry` instances. Modern molecular simulation workflows
often involve high volumes of data -- anywhere between gigabytes and terabytes.
The use of regular Python lists to keep track of all atomic data would induce
excessive (and unnecessary) RAM consumption.
To overcome this, psiflow `Dataset` instances keep track of data on disk, in a
human-readable (extended) XYZ format.

In combination with the concept of futures, psiflow datasets can represent data that is
either currently available (e.g. from a user-provided XYZ file) or data _that will be
generated in the future_, for example the trajectory of a molecular dynamics simulation.

Practically speaking, `Dataset` instances behave like a regular list of geometries:

```py
from psiflow.data import Dataset

data = Dataset.load('trajectory.xyz')     # data is of type Dataset

data[4]           # AppFuture representing the `Geometry` instance at index 4
data.length()     # AppFuture representing the length of the dataset

data.geometries() # AppFuture representing a list of Geometry instances

```
As shown in the example, you can still index the dataset and ask for its length as you would normally do when working directly with a Python list.
The only difference is that it returns futures instead of geometries when asking for any
given element, and returns a future of an integer instead of an integer when asking for
the length.

```py
print(data[4].result())         #  actual `Geometry` instance at index 4
print(data.length().result())   #  actual length of the dataset, i.e. the number of states in `train.xyz`
```

In addition, slicing a `Dataset` returns a new `Dataset`, in which the extracted data is
or will be copied.
```py
data[-5:]         # Dataset which contains the last five structures
```

Datasets do not support item assignment, i.e. `data[5] = geometry` will not work.

In addition to list-like functionality, `Dataset` provides a number of convenience methods for common operations such as filtering, shuffling, or creating a training/validation split.
```py
train, valid = data.split(0.9, shuffle=True)  # do a randomized 90/10 split

energies = train.get('energy')                # get the energies of all geometries in the training set
print(energies.result().shape)                # energies is an AppFuture, so we need to call .result()
# (n,)

forces = train.get('forces')                  # get the forces of all geometries in the training set
print(forces.result().shape)                  # forces is an AppFuture, so we need to call .result()
# (n, 3)
```
