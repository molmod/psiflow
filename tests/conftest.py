from pathlib import Path

import ase
import numpy as np
import parsl
import pytest
import yaml
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.calculators.emt import EMT

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.models import MACE
from psiflow.hamiltonians import MACEHamiltonian


def pytest_addoption(parser):
    parser.addoption(
        "--psiflow-config",
        action="store",
        help="test",
    )
    parser.addoption(
        "--skip-gpu",
        action="store_true",
        default=False,
        help="whether to run tests which require a GPU",
    )


@pytest.fixture(scope="session")
def gpu(request):
    if request.config.getoption("--skip-gpu"):
        pytest.skip("skipping tests which require GPU")


@pytest.fixture(scope="session", autouse=True)
def context(request, tmp_path_factory):
    try:
        context = psiflow.context()
    except RuntimeError:
        path_config = Path(request.config.getoption("--psiflow-config"))
        with open(path_config, "r") as f:
            psiflow_config = yaml.safe_load(f)
        psiflow.load(psiflow_config)
        context = psiflow.context()  # noqa: F841
        yield
        parsl.dfk().cleanup()


@pytest.fixture(scope="session")
def mace_config() -> dict:
    return dict(
        num_radial_basis=3,
        num_cutoff_basis=2,
        max_ell=1,
        correlation=1,
        MLP_irreps="2x0e",
        num_channels=2,
        max_L=0,
        r_max=4,
        radial_MLP="[4]",
        swa=True,
        #
        batch_size=1,  # MACE crashes if it is larger than dataset size
        max_num_epochs=10,
        energy_weight=100,  # make sure energy RMSE drops from first epoch
        forces_weight=10
    )


def generate_emt_cu_data(nstates, amplitude, supercell=None) -> list[ase.Atoms]:
    if supercell is None:
        supercell = np.eye(3)
    atoms = make_supercell(bulk("Cu", "fcc", a=3.6, cubic=True), supercell)
    atoms.calc = EMT()
    pos = atoms.get_positions()
    box = atoms.get_cell()
    atoms_list = []
    for _ in range(nstates):
        atoms.set_positions(
            pos + np.random.uniform(-amplitude, amplitude, size=(len(atoms), 3))
        )
        atoms.set_cell(box + np.random.uniform(-amplitude, amplitude, size=(3, 3)))
        _atoms = atoms.copy()
        _atoms.calc = None
        _atoms.info["energy"] = atoms.get_potential_energy()
        _atoms.info["stress"] = atoms.get_stress(voigt=False)
        _atoms.arrays["forces"] = atoms.get_forces()
        # make content heterogeneous to test per_element functions
        _atoms.numbers[0] = 1
        _atoms.symbols[0] = "H"
        atoms_list.append(_atoms)
    return atoms_list


@pytest.fixture
def dataset(context) -> Dataset:
    data = generate_emt_cu_data(20, 0.2)
    data += generate_emt_cu_data(5, 0.15, supercell=np.diag([1, 2, 1]))
    data_ = [Geometry.from_atoms(atoms) for atoms in data]
    return Dataset(data_).align_axes()


# @pytest.fixture(scope="session")
# def mace_model(tmp_path_factory, mace_config):
#     path = tmp_path_factory.mktemp("mace")
#     model = MACEModel(path, mace_config)
#     # manually recreate dataset with 'session' scope
#     data = generate_emt_cu_data(20, 0.2)
#     data_ = [Geometry.from_atoms(atoms) for atoms in data]
#     dataset = Dataset(data_)
#     # add additional state to initialize other atomic numbers
#     geometry = Geometry.from_data(
#         numbers=np.array(2 * [101]),
#         positions=np.array([[0, 0, 0], [2, 0, 0]]),
#         cell=None,
#     )
#     geometry.energy = -1.0
#     geometry.per_atom.forces[:] = np.random.uniform(size=(2, 3))
#     model.initialize(dataset[:5] + Dataset([geometry]))
#     return model


@pytest.fixture(scope="session")
def mace_foundation() -> Path:
    hamiltonian = MACEHamiltonian.from_foundation(name="small")
    return hamiltonian.external.filepath


@pytest.fixture
def dataset_h2(context):
    h2 = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0.74, 0, 0]], pbc=False)
    data = [h2.copy() for i in range(20)]
    for atoms in data:
        atoms.positions += np.random.uniform(-0.05, 0.05, size=(2, 3))
    return Dataset([Geometry.from_atoms(a) for a in data])


@pytest.fixture
def checkpoint() -> str:
    return """
<simulation>
   <output prefix='output'>
      <checkpoint filename='checkpoint' stride='10'>1</checkpoint>
      <properties shape='(3)' filename='properties' stride='10'> [ time, temperature, potential ] </properties>
   </output>
   <total_steps>100</total_steps>
   <ffsocket mode='unix' name='EinsteinCrystal0'>
      <address>cSzwsJ2A/einsteincrystal0</address>
      <timeout>  8.33333333e-02</timeout>
   </ffsocket>
   <ffsocket mode='unix' name='PlumedHamiltonian0'>
      <address>cSzwsJ2A/plumedhamiltonian0</address>
      <timeout>  8.33333333e-02</timeout>
   </ffsocket>
   <system prefix='walker-0'>
      <forces>
         <force forcefield='EinsteinCrystal0'>
         </force>
         <force forcefield='PlumedHamiltonian0'>
         </force>
      </forces>
      <ensemble>
         <temperature>  1.90008912e-03</temperature>
         <eens>  4.11423554e-03</eens>
         <hamiltonian_weights shape='(1)'> [   1.00000000e+00 ] </hamiltonian_weights>
         <time>  2.06706865e+02</time>
      </ensemble>
      <motion mode='dynamics'>
         <dynamics mode='nvt'>
            <thermostat mode='langevin'>
               <tau>  4.13413730e+03</tau>
            </thermostat>
            <timestep>  2.06706865e+01</timestep>
            <nmts shape='(1)'> [ 1 ] </nmts>
         </dynamics>
      </motion>
      <beads natoms='4' nbeads='1'>
         <q shape='(1, 12)'>
          [   1.44513572e-01,  -2.22608601e-02,   6.90340566e-02,  -1.48068714e-01,   3.67026570e+00,
              3.24415892e+00,   3.09455639e+00,  -2.66306646e-01,   3.36282329e+00,   3.54200180e+00,
              3.39685661e+00,   5.46722856e-01 ]
         </q>
         <p shape='(1, 12)'>
          [   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00 ]
         </p>
         <m shape='(4)'> [   1.83747161e+03,   1.15837273e+05,   1.15837273e+05,   1.15837273e+05 ] </m>
         <names shape='(4)'> [ H, Cu, Cu, Cu ] </names>
      </beads>
      <cell shape='(3, 3)'>
       [   1e+00,   1e-01,  0,   0.00000000e+00,   2e+00,
          0,   0.00000000e+00,   0.00000000e+00,   3e+00 ]
      </cell>
   </system>
   <system prefix='walker-1'>
      <forces>
         <force forcefield='EinsteinCrystal0'>
         </force>
         <force weight='  0.00000000e+00' forcefield='PlumedHamiltonian0'>
         </force>
      </forces>
      <ensemble>
         <temperature>  1.90008912e-03</temperature>
         <eens>  4.11423554e-03</eens>
         <hamiltonian_weights shape='(1)'> [   1.00000000e+00 ] </hamiltonian_weights>
         <time>  2.06706865e+02</time>
      </ensemble>
      <motion mode='dynamics'>
         <dynamics mode='nvt'>
            <thermostat mode='langevin'>
               <tau>  4.13413730e+03</tau>
            </thermostat>
            <timestep>  2.06706865e+01</timestep>
            <nmts shape='(1)'> [ 1 ] </nmts>
         </dynamics>
      </motion>
      <beads natoms='4' nbeads='1'>
         <q shape='(1, 12)'>
          [   1.44513572e-01,  -2.22608601e-02,   6.90340566e-02,  -1.48068714e-01,   3.67026570e+00,
              3.24415892e+00,   3.09455639e+00,  -2.66306646e-01,   3.36282329e+00,   3.54200180e+00,
              3.39685661e+00,   5.46722856e-01 ]
         </q>
         <p shape='(1, 12)'>
          [   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00 ]
         </p>
         <m shape='(4)'> [   1.83747161e+03,   1.15837273e+05,   1.15837273e+05,   1.15837273e+05 ] </m>
         <names shape='(4)'> [ H, Cu, Cu, Cu ] </names>
      </beads>
      <cell shape='(3, 3)'>
       [   6.92067797e+00,   1.35926184e-01,  -3.29542567e-02,   0.00000000e+00,   6.46614176e+00,
          -3.74701247e-01,   0.00000000e+00,   0.00000000e+00,   6.45073059e+00 ]
      </cell>
   </system>
</simulation>
"""
