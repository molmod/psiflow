import numpy as np
import requests

from ase import Atoms

from pymatgen.io.cp2k.inputs import Cp2kInput

from autolearn.reference import EMTReference, CP2KReference
from autolearn import ReferenceExecution, Sample
from autolearn.reference._cp2k import insert_filepaths_in_input, \
        insert_atoms_in_input

from utils import generate_emt_cu_data


sample_input = """
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


def test_reference_emt(tmp_path):
    atoms_list = generate_emt_cu_data(a=3.6, nstates=1)
    atoms = atoms_list[0]
    e0 = atoms.info.pop('energy')

    reference = EMTReference() # redo computation via EMTReference
    reference_execution = ReferenceExecution()
    sample = Sample(atoms)
    assert not sample.evaluated
    assert len(sample.tags) == 0
    sample = reference.evaluate(sample, reference_execution)
    assert np.allclose(e0, sample.atoms.info['energy'])


def test_cp2k_insert_filepaths(tmp_path):
    filepaths = {
            'BASIS_SET_FILE_NAME': ['basisset0', 'basisset1'],
            'POTENTIAL_FILE_NAME': 'potential',
            'PARAMETER_FILE_NAME': 'parameter',
            }
    target_input = """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  basisset0
      BASIS_SET_FILE_NAME  basisset1
      POTENTIAL_FILE_NAME  potential
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
    target = Cp2kInput.from_string(target_input)
    sample = Cp2kInput.from_string(insert_filepaths_in_input(sample_input, filepaths))
    assert str(target) == str(sample)


def test_cp2k_insert_atoms(tmp_path):
    atoms = Atoms(numbers=np.ones(3), cell=np.eye(3), positions=np.eye(3), pbc=True)
    sample = Cp2kInput.from_string(insert_atoms_in_input(sample_input, atoms))
    assert 'COORD' in sample['FORCE_EVAL']['SUBSYS'].subsections.keys()
    assert 'CELL' in sample['FORCE_EVAL']['SUBSYS'].subsections.keys()
    natoms = len(sample['FORCE_EVAL']['SUBSYS']['COORD'].keywords['H'])
    assert natoms == 3


def test_cp2k_success(tmp_path):
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    data = {
            'BASIS_SET_FILE_NAME': basis,
            'POTENTIAL_FILE_NAME': potential,
            'PARAMETER_FILE_NAME': dftd3,
            }
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
         BASIS_SET DZVP-MOLOPT-PBE-GTH-q1
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
"""
    atoms = Atoms(
            numbers=np.ones(2),
            cell=3.5 * np.eye(3),
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            pbc=True,
            )
    reference_execution = ReferenceExecution(
            ncores=4, # avoid double counting due to HyperT
            mpi=lambda x: ['mpirun', f' -np {x}'],
            )
    reference = CP2KReference(cp2k_input, data)
    sample = reference.evaluate(
            Sample(atoms),
            reference_execution,
            )
    assert 'success' in sample.tags
    assert sample.evaluated
    assert 'energy' in sample.atoms.info.keys()
    assert 'stress' in sample.atoms.info.keys()
    assert 'forces' in sample.atoms.arrays.keys()
    assert sample.log is not None


def test_cp2k_failure(tmp_path):
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    data = {
            'BASIS_SET_FILE_NAME': basis,
            'POTENTIAL_FILE_NAME': potential,
            'PARAMETER_FILE_NAME': dftd3,
            }
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
""" # incorrect input file
    atoms = Atoms(
            numbers=np.ones(2),
            cell=3.5 * np.eye(3),
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            pbc=True,
            )
    reference_execution = ReferenceExecution(
            ncores=4, # avoid double counting due to HyperT
            mpi=lambda x: ['mpirun', f' -np {x}'],
            )
    reference = CP2KReference(cp2k_input, data)
    sample = reference.evaluate(
            Sample(atoms),
            reference_execution,
            )
    assert 'error' in sample.tags
    assert sample.log is not None


def test_cp2k_timeout(tmp_path):
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    data = {
            'BASIS_SET_FILE_NAME': basis,
            'POTENTIAL_FILE_NAME': potential,
            'PARAMETER_FILE_NAME': dftd3,
            }
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
         BASIS_SET DZVP-MOLOPT-PBE-GTH-q1
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
"""
    atoms = Atoms(
            numbers=np.ones(2),
            cell=3.5 * np.eye(3),
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            pbc=True,
            )
    reference_execution = ReferenceExecution(
            ncores=4, # avoid double counting due to HyperT
            mpi=lambda x: ['mpirun', f' -np {x}'],
            walltime=10,
            )
    reference = CP2KReference(cp2k_input, data)
    sample = reference.evaluate(
            Sample(atoms),
            reference_execution,
            )
    assert 'error' in sample.tags
    assert 'timeout' in sample.tags
    assert not sample.evaluated
    assert sample.log is not None
