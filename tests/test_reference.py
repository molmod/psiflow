from pathlib import Path

import molmod
import numpy as np
import pytest
import requests
from ase import Atoms
from ase.units import Bohr, Ha, Pascal
from parsl.dataflow.futures import AppFuture
from pymatgen.io.cp2k.inputs import Cp2kInput

import psiflow
from psiflow.data import Dataset, FlowAtoms, NullState
from psiflow.reference import CP2KReference, EMTReference, PySCFReference
from psiflow.reference._cp2k import insert_filepaths_in_input
from psiflow.reference._pyscf import generate_script, parse_energy_forces


@pytest.fixture
def fake_cp2k_input():
    return """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/BASISSETS
      POTENTIAL_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/GTH_POTENTIALS
      &XC
         &VDW_POTENTIAL
            &PAIR_POTENTIAL
               PARAMETER_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/dftd3.dat
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND Al
         ELEMENT  H
         BASIS_SET foo
         POTENTIAL bar
      &END KIND
      &COORD
         H 4.0 0.0 0.0
      &END COORD
    &END SUBSYS
&END FORCE_EVAL
"""


@pytest.fixture
def cp2k_data():
    basis = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH"
    ).text
    dftd3 = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat"
    ).text
    potential = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH"
    ).text
    return {
        "basis_set": basis,
        "potential": potential,
        "dftd3": dftd3,
    }


@pytest.fixture
def cp2k_input():
    # remove stress tensor keyword to ensure it gets added automatically
    return """
&FORCE_EVAL
   METHOD Quickstep
   !STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  dummy
      POTENTIAL_FILE_NAME  dummy
      &SCF
         MAX_SCF  10
         MAX_DIIS  8
         EPS_SCF  1.0E-06
         SCF_GUESS  RESTART
         &OT
            MINIMIZER  CG
            PRECONDITIONER  FULL_SINGLE_INVERSE
         &END OT
         &OUTER_SCF T
            MAX_SCF  10
            EPS_SCF  1.0E-06
         &END OUTER_SCF
      &END SCF
      &QS
         METHOD  GPW
         EPS_DEFAULT  1.0E-4
         EXTRAPOLATION  USE_GUESS
      &END QS
      &MGRID
         REL_CUTOFF [Ry]  60.0
         NGRIDS  5
         CUTOFF [Ry] 1000
      &END MGRID
      &XC
         DENSITY_CUTOFF   1.0E-10
         GRADIENT_CUTOFF  1.0E-10
         TAU_CUTOFF       1.0E-10
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
         &VDW_POTENTIAL
            POTENTIAL_TYPE  PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE  DFTD3(BJ)
               PARAMETER_FILE_NAME  parameter
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF  25
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND H
         ELEMENT  H
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q1
         POTENTIAL GTH-PBE-q1
      &END KIND
      &KIND O
         ELEMENT  O
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q6
         POTENTIAL GTH-PBE-q6
      &END KIND
      &KIND Si
         ELEMENT  Si
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q4
         POTENTIAL GTH-PBE-q4
      &END KIND
      &KIND C
         ELEMENT  C
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q4
         POTENTIAL GTH-PBE-q4
      &END KIND
      &KIND Al
         ELEMENT  Al
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q3
         POTENTIAL GTH-PBE-q3
      &END KIND
   &END SUBSYS
!   &PRINT
!      &STRESS_TENSOR ON
!      &END STRESS_TENSOR
!      &FORCES
!      &END FORCES
!   &END PRINT
&END FORCE_EVAL
"""


@pytest.fixture
def cp2k_reference(context, cp2k_input, cp2k_data, tmp_path):
    reference = CP2KReference(cp2k_input=cp2k_input)
    for key, value in cp2k_data.items():
        with open(tmp_path / key, "w") as f:
            f.write(value)
        reference.add_file(key, tmp_path / key)
    return reference


def test_reference_emt(context, dataset, tmp_path):
    reference = EMTReference()
    # modify dataset to include states for which EMT fails:
    _ = reference.evaluate(dataset).as_list().result()
    atoms_list = dataset.as_list().result()
    atoms_list[6].numbers[1] = 90
    atoms_list[9].numbers[1] = 3
    dataset_ = Dataset(atoms_list)
    evaluated = reference.evaluate(dataset_)
    assert evaluated.length().result() == len(atoms_list)

    atoms = reference.evaluate(dataset_[5]).result()
    assert type(atoms) is FlowAtoms
    assert atoms.reference_status
    atoms = reference.evaluate(dataset_[6]).result()
    assert type(atoms) is FlowAtoms
    assert not atoms.reference_status


def test_cp2k_insert_filepaths(fake_cp2k_input):
    filepaths = {
        "basis_set": "basisset0",
        "basis_giggle": "basisset1",
        "potential": "potential",
        "dftd3": "parameter",
    }
    target_input = """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      POTENTIAL_FILE_NAME  potential
      BASIS_SET_FILE_NAME  basisset0
      BASIS_SET_FILE_NAME  basisset1
      &XC
         &VDW_POTENTIAL
            &PAIR_POTENTIAL
               PARAMETER_FILE_NAME  parameter
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND Al
         ELEMENT  H
         BASIS_SET foo
         POTENTIAL bar
      &END KIND
      &COORD
         H 4.0 0.0 0.0
      &END COORD
    &END SUBSYS
&END FORCE_EVAL
"""
    target = Cp2kInput.from_str(target_input)
    sample = Cp2kInput.from_str(insert_filepaths_in_input(fake_cp2k_input, filepaths))
    assert str(target) == str(sample)


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_success(context, cp2k_reference):
    atoms = FlowAtoms(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        cell=5 * np.eye(3),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        pbc=True,
    )
    dataset = Dataset([atoms])
    evaluated = cp2k_reference.evaluate(dataset[0])
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().reference_status
    assert Path(evaluated.result().reference_stdout).is_file()
    assert Path(evaluated.result().reference_stderr).is_file()
    assert "energy" in evaluated.result().info.keys()
    assert "stress" in evaluated.result().info.keys()
    assert "forces" in evaluated.result().arrays.keys()
    assert np.allclose(
        -1.165271567241256 / molmod.units.electronvolt,
        evaluated.result().info["energy"],
    )
    forces_reference = np.array(
        [[-0.01215748, 0.00001210, 0.00001210], [0.01217855, 0.00001150, 0.00001150]]
    )
    forces_reference /= molmod.units.electronvolt
    forces_reference *= molmod.units.angstrom
    assert np.allclose(
        forces_reference,
        evaluated.result().arrays["forces"],
        atol=1e-5,
    )
    stress_reference = -1.0 * np.array(
        [
            [4.81505171868e-01, 4.49529611310e-06, 4.49529611348e-06],
            [4.49529611310e-06, -9.53484935396e-03, 1.47299106211e-04],
            [4.49529611348e-06, 1.47299106211e-04, -9.53484935396e-03],
        ]
    )
    stress_reference *= 1e9 * Pascal
    assert np.allclose(
        stress_reference,
        evaluated.result().info["stress"],
        # atol=1e-5,
    )

    # check whether NullState evaluates to NullState
    state = cp2k_reference.evaluate(NullState)
    assert state.result() == NullState

    # check number of mpi processes
    with open(evaluated.result().reference_stdout, "r") as f:
        content = f.read()
    context = psiflow.context()
    ncores = context[CP2KReference].cores_per_worker
    lines = content.split("\n")
    for line in lines:
        if "Total number of message passing processes" in line:
            nprocesses = int(line.split()[-1])
        if "Number of threads for this process" in line:
            nthreads = int(line.split()[-1])
    assert ncores == nprocesses
    assert 1 == nthreads


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_failure(context, cp2k_data, tmp_path):
    cp2k_input = """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  dummy
      POTENTIAL_FILE_NAME  dummy
      &SCF
         MAX_SCF  10
         MAX_DIIS  8
         EPS_SCF  1.0E-01
         SCF_GUESS  RESTART
         &OT
            MINIMIZER  CG
            PRECONDITIONER  FULL_SINGLE_INVERSE
         &END OT
         &OUTER_SCF T
            MAX_SCF  10
            EPS_SCF  1.0E-01
         &END OUTER_SCF
      &END SCF
      &QS
         METHOD  GPW
         EPS_DEFAULT  1.0E-4
         EXTRAPOLATION  USE_GUESS
      &END QS
      &MGRID
         REL_CUTOFF [Ry]  60.0
         NGRIDS  5
         CUTOFF [Ry] 200
      &END MGRID
      &XC
         DENSITY_CUTOFF   1.0E-10
         GRADIENT_CUTOFF  1.0E-10
         TAU_CUTOFF       1.0E-10
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
         &VDW_POTENTIAL
            POTENTIAL_TYPE  PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE  DFTD3(BJ)
               PARAMETER_FILE_NAME  parameter
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF  25
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND H
         ELEMENT  H
         BASIS_SET XXXXXXXXXX
         POTENTIAL GTH-PBE-q1
      &END KIND
   &END SUBSYS
   &PRINT
      &STRESS_TENSOR ON
      &END STRESS_TENSOR
      &FORCES
      &END FORCES
   &END PRINT
&END FORCE_EVAL
"""  # incorrect input file
    reference = CP2KReference(cp2k_input=cp2k_input)
    for key, value in cp2k_data.items():
        with open(tmp_path / key, "w") as f:
            f.write(value)
        reference.add_file(key, tmp_path / key)
    atoms = FlowAtoms(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        cell=5 * np.eye(3),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        pbc=True,
    )
    evaluated = reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert not evaluated.result().reference_status
    assert "energy" not in evaluated.result().info.keys()
    with open(evaluated.result().reference_stdout, "r") as f:
        log = f.read()
    assert "ABORT" in log  # verify error is captured
    assert "requested basis set" in log


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_timeout(context, cp2k_reference):
    atoms = FlowAtoms(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        cell=20 * np.eye(3),  # box way too large
        positions=np.array([[0, 0, 0], [3, 0, 0]]),
        pbc=True,
    )
    evaluated = cp2k_reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert not evaluated.result().reference_status
    assert "energy" not in evaluated.result().info.keys()


def test_emt_atomic_energies(context, dataset):
    reference = EMTReference()
    for element in ["H", "Cu"]:
        energy = reference.compute_atomic_energy(element, box_size=5)
        energy_ = reference.compute_atomic_energy(element, box_size=7)
        assert energy.result() < energy_.result()


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_atomic_energies(cp2k_reference, dataset):
    element = "H"
    energy = cp2k_reference.compute_atomic_energy(element, box_size=4)
    assert abs(energy.result() - (-13.6)) < 1  # reasonably close to exact value


# @pytest.fixture
# def nwchem_reference(context):
#    calculator_kwargs = {
#        "basis": {"H": "cc-pvqz"},
#        "dft": {
#            "xc": "pbe96",
#            "mult": 1,
#            "convergence": {
#                "energy": 1e-6,
#                "density": 1e-6,
#                "gradient": 1e-6,
#            },
#            "disp": {"vdw": 3},
#        },
#    }
#    return NWChemReference(**calculator_kwargs)
#
#
# def test_nwchem_success(nwchem_reference):
#    atoms = FlowAtoms(  # simple H2 at ~optimized interatomic distance
#        numbers=np.ones(2),
#        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
#    )
#    dataset = Dataset([atoms])
#    evaluated = nwchem_reference.evaluate(dataset[0])
#    assert isinstance(evaluated, AppFuture)
#    assert evaluated.result().reference_status
#    assert Path(evaluated.result().reference_stdout).is_file()
#    assert Path(evaluated.result().reference_stderr).is_file()
#    assert "energy" in evaluated.result().info.keys()
#    assert "stress" not in evaluated.result().info.keys()
#    assert "forces" in evaluated.result().arrays.keys()
#    assert evaluated.result().arrays["forces"][0, 0] < 0
#    assert evaluated.result().arrays["forces"][1, 0] > 0
#
#    nwchem_reference.evaluate(dataset)
#    assert nwchem_reference.compute_atomic_energy("H").result() < 0


@pytest.fixture
def pyscf_reference(context):
    routine = """
from pyscf import dft

mf = dft.RKS(molecule)
mf.xc = 'pbe,pbe'

energy = mf.kernel()
forces = -mf.nuc_grad_method().kernel()
"""
    basis = "cc-pvtz"
    spin = 0
    return PySCFReference(routine, basis, spin)


def test_pyscf_generate_script(pyscf_reference):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], pbc=False)
    _ = generate_script(
        atoms,
        pyscf_reference.routine,
        pyscf_reference.basis,
        pyscf_reference.spin,
    )


def test_pyscf_extract_energy_forces():
    stdout = """
total energy = as;ldkfj
tion, you can put the setting "B3LYP_WITH_VWN5 = True" in pyscf_conf.py
  warnings.warn('Since PySCF-2.3, B3LYP (and B3P86) are changed to the VWN-RPA variant, '
converged SCF energy = -1.16614771756639
total forces = ;s';dlfkj
--------------- RKS gradients ---------------
         x                y                z
0 H    -0.0000000000     0.0000000000     0.0005538080
1 H    -0.0000000000     0.0000000000    -0.0005538080
----------------------------------------------
total energy = -31.73249570413387
total forces =
8.504463377857879e-16 -6.70273006321553e-16 -0.02847795118636264
1.9587146782865252e-16 -2.135019926691156e-15 0.028477951186359783
"""
    energy, forces = parse_energy_forces(stdout)
    assert np.allclose(energy, -1.16614771756639 * Ha)
    forces_ = (
        (-1.0)
        * np.array(
            [
                [-0.0000000000, 0.0000000000, 0.0005538080],
                [-0.0000000000, 0.0000000000, -0.0005538080],
            ]
        )
        * Ha
        / Bohr
    )
    assert np.allclose(forces, forces_)


def test_pyscf_success(pyscf_reference):
    atoms = FlowAtoms(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
    )
    dataset = Dataset([atoms])
    evaluated = pyscf_reference.evaluate(dataset[0])
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().reference_status
    assert Path(evaluated.result().reference_stdout).is_file()
    assert Path(evaluated.result().reference_stderr).is_file()
    assert "energy" in evaluated.result().info.keys()
    assert "stress" not in evaluated.result().info.keys()
    assert "forces" in evaluated.result().arrays.keys()
    assert evaluated.result().arrays["forces"][0, 0] < 0
    assert evaluated.result().arrays["forces"][1, 0] > 0

    pyscf_reference.evaluate(dataset)
    assert pyscf_reference.compute_atomic_energy("H").result() < 0


def test_pyscf_timeout(context):
    routine = """
from pyscf import scf, cc

mf = scf.HF(molecule).run()
mycc = cc.CCSD(mf)
mycc.kernel()
energy = mycc.e_tot
forces = -mycc.nuc_grad_method().kernel()
"""
    basis = "cc-pvqz"
    spin = 0
    reference = PySCFReference(routine, basis, spin)
    atoms = FlowAtoms(
        numbers=np.ones(4),
        positions=np.array([[0, 0, 0], [0.74, 0, 0], [0, 3, 0], [0.74, 3, 0]]),
    )
    reference.evaluate(atoms).result()
    assert not atoms.reference_status
