import os
import molmod
import yaff
import numpy as np

from ase.geometry import Cell
from ase import Atoms


class ForceThresholdExceededException(Exception):
    pass


class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms, force_threshold):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        force_threshold : float [eV/A]

        """
        yaff.pes.ForcePart.__init__(self, 'ase', system)
        self.system = system # store system to obtain current pos and box
        self.atoms  = atoms
        self.force_threshold = force_threshold

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            self.check_threshold(forces)
            gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
        if vtens is not None:
            try: # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            volume = np.linalg.det(self.atoms.get_cell())
            vtens[:] = volume * stress * molmod.units.electronvolt
        return energy

    def check_threshold(self, forces):
        max_force = np.max(np.linalg.norm(forces, axis=1))
        index = np.argmax(np.linalg.norm(forces, axis=1))
        if max_force > self.force_threshold:
            raise ForceThresholdExceededException(
                    'Max force exceeded: {} eV/A by atom index {}'.format(max_force, index),
                    )


def create_forcefield(atoms, force_threshold):
    """Creates force field from ASE atoms instance"""
    system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )
    system.set_standard_masses()
    part_ase = ForcePartASE(system, atoms, force_threshold)
    return yaff.pes.ForceField(system, [part_ase])


class DataHook(yaff.VerletHook):

    def __init__(self, start=0, step=1):
        super().__init__(start, step)
        self.atoms = None
        self.data = []

    def init(self, iterative):
        self.atoms = Atoms(
                numbers=iterative.ff.system.numbers.copy(),
                positions=iterative.ff.system.pos / molmod.units.angstrom,
                cell=iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                pbc=True,
                )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom)
        self.data.append(self.atoms.copy())


class ExtXYZHook(yaff.VerletHook): # xyz file writer; obsolete

    def __init__(self, path_xyz, start=0, step=1):
        super().__init__(start, step)
        Path(path_xyz).unlink(missing_ok=True) # remove if exists
        self.path_xyz = path_xyz
        self.atoms = None

    def init(self, iterative):
        self.atoms = Atoms(
                numbers=iterative.ff.system.numbers.copy(),
                positions=iterative.ff.system.pos / molmod.units.angstrom,
                cell=iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                pbc=True,
                )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom)
        write(self.path_xyz, self.atoms, append=True)


def try_manual_plumed_linking():
    if 'PLUMED_KERNEL' not in os.environ.keys():
        # try linking manually
        if 'CONDA_PREFIX' in os.environ.keys(): # for conda environments
            p = 'CONDA_PREFIX'
        elif 'PREFIX' in os.environ.keys(): # for pip environments
            p = 'PREFIX'
        else:
            print('failed to set plumed .so kernel')
            pass
        path = os.environ[p] + '/lib/libplumedKernel.so'
        if os.path.exists(path):
            os.environ['PLUMED_KERNEL'] = path
            print('plumed kernel manually set at at : {}'.format(path))


def apply_strain(strain, box0):
    """Applies a strain tensor to a reference box

    The resulting strained box matrix is obtained based on:

        box = box0 @ sqrt(2 * strain + I)

    where the second argument is computed based on a diagonalization of
    2 * strain + I.

    Parameters
    ----------

    strain : ndarray of shape (3, 3)
        desired strain matrix

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    assert np.allclose(strain, strain.T)
    A = 2 * strain + np.eye(3)
    values, vectors = np.linalg.eigh(A)
    sqrtA = vectors @ np.sqrt(np.diag(values)) @ vectors.T
    box = box0 @ sqrtA
    return box


def compute_strain(box, box0):
    """Computes the strain of a given box with respect to a reference

    The strain matrix is defined by the following expression

        strain = 0.5 * (inv(box0) @ box @ box.T @ inv(box0).T - I)

    Parameters
    ----------

    box : ndarray of shape (3, 3)
        box matrix for which to compute the strain

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    box0inv = np.linalg.inv(box0)
    return 0.5 * (box0inv @ box @ box.T @ box0inv.T - np.eye(3))
