# QM calculations
The energy and gradients of the ground-state Born-Oppenheimer surface can be obtained
using varying levels of approximation.
In psiflow, the calculation of the energy and its gradients can be performed for both
`Geometry` and `Dataset` instances, using different software packages:

- **CP2K** (periodic, mixed PW/lcao): very fast, and very useful for pretty much any periodic
  structure. Its forces tend to be quite noisy with the default grid settings so some
  level of caution is advised. Also, even though it uses both plane waves and atomic basis
  sets, it does suffer from BSSE.
- **GPAW** (periodic/cluster, PW/lcao/grid): slower but more numerically stable than CP2K;
  essentially a fully open-source (and therefore transparant), free, and well-tested
  alternative to VASP. Particularly useful for applications in which BSSE is a concern
  (e.g. adsorption).
- **ORCA** (cluster, lcao): useful for accurate high-level quantum chemistry calculations,
  e.g. MP2 and CCSD(T). *TODO*

!!! note "Installation"
    Because the 'correct' compilation and installation of quantum chemistry software is
    notoriously cumbersome, we host separate container images for each of the packages
    on Github, which are ready to use with psiflow on HPCs with either a Singularity
    or Apptainer container runtime. The Docker files used to generate those images are
    available in the respository; 
    [CP2K](https://github.com/molmod/psiflow/blob/main/Dockerfile.cp2k) or
    [GPAW](https://github.com/molmod/psiflow/blob/main/Dockerfile.gpaw).
    See the [configuration](configuration.md) section for more details.

For each software package, psiflow provides a corresponding class which implements
the appropriate input file manipulations, launch commands, and output parsing
functionalities.
They all inherit from the `Reference` base class, which provides a few key
functionalities:

- `data.evaluate(reference)`: this is the most common operation involving QM calculations;
  given a `Dataset` of atomic geometries, compute the energy and its gradients and insert
  them into the dataset such that they are saved for future reuse.
- `reference.compute_atomic_energy`: provides the ability to compute isolated atom
  reference energies, as this facilitates ML potential training to datasets with varying
  number of atoms.
- `reference.compute(data)`: this is somewhat equivalent to the hamiltonian `compute`
  method, except that its argument `data` must be a `Dataset` instance, and the optional
  `batch_size` defaults to 1 (in order to maximize parallelization). It does not insert
  the computed properties into the data, but returns them as numpy arrays.

From a distance, QM reference objects look almost identical to hamiltonians, in the sense
that they both take atomic geometries as input and return energies and gradients as
output. The (imposed) distinction between both can be summarized in the following points.

- hamiltonians can compute energies and forces for pretty much *any* structure. There is
  no reason they would fail. QM calculations on the other hand can fail due to unconverged
  SCF cycles and/or time limit constraints. In fact, this happens relatively often when performing
  active learning workflows. Reference objects take this into account by returning a unique
  `NullState` whenever a calculation has failed.
- hamiltonians are orders of magnitude faster, and can be employed in meaningfully long
  molecular dynamics simulations. This is not the case for QM calculations. As such, they
  cannot be used in combination with walker sampling or geometry optimizations. If the
  purpose is to perform molecular simulation at the DFT level, then the better approach is
  to train an ML potential to any desired level of accuracy (almost always possible in
  psiflow) and use that as proxy for the QM interaction energy.
  For the same reason, the default batch size for `reference.compute` calls is 1, i.e.
  each the QM calculation for each structure in the dataset is immediately scheduled
  independently from the other ones.
  With hamiltonians, that batch size defaults to 100 (split data in chunks of 100 and
  evaluate each set of 100 states serially).


## CP2K 2024.1
A `CP2K` reference instance can be created based on a (multiline) input string.
Only the `FORCE_EVAL` section of the input is important since the atomic coordinates and cell
parameters are automatically inserted for every calculation.
All basis set, pseudopotential, and D3 parameters from the official
[CP2K repository](https://github.com/cp2k/cp2k) are directly available in the
container image (i.e. no need to download or provide these files separately).
Choose which one you would like to use by using the corresponding filename in the input
file (i.e. omit any preceding filepaths).
A typical [input file](https://github.com/molmod/psiflow/blob/main/examples/data/cp2k_input.txt)
is provided in the [examples](https://github.com/molmod/psiflow/tree/main/examples).

```py
from psiflow.reference import CP2K


# create reference instance
with open('cp2k_input.txt', 'r') as f:
    force_eval_input_str = f.read()
cp2k = CP2K(force_eval_input_str)

# compute energy and forces, and store them in the geometries
evaluated_data = data.evaluate(cp2k)

for geometry in evaluated_data.geometries().result():
    print('energy: {} eV'.format(geometry.energy))
    print('forces: {} eV/A'.format(geometry.per_atom.forces))

```

## GPAW 24.1
a `GPAW` reference is created in much the same way as a traditional `GPAW` 'calculator'
instance, with support for entirely the same keyword arguments:
```py
from psiflow.reference import GPAW

gpaw = GPAW(mode='fd', nbands=0, xc='PBE')  # see GPAW calculator on gitlab for full list
energies = gpaw.compute(data, 'energy')

```
A notable feature from GPAW is that it already outputs all energies as formation energies,
i.e. it internally subtracts the sum of the energies of the isolated atoms. As such, the
`compute_atomic_energy` for a GPAW reference always just returns 0 eV.

## ORCA
TODO
