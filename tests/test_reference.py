import requests
import pytest
import os
import molmod
import numpy as np
from parsl.dataflow.futures import AppFuture
from parsl.app.futures import DataFuture

from pymatgen.io.cp2k.inputs import Cp2kInput

from ase import Atoms

from psiflow.data import FlowAtoms, parse_evaluation_logs
from psiflow.reference import EMTReference, CP2KReference
from psiflow.reference._cp2k import insert_filepaths_in_input, \
        insert_atoms_in_input
from psiflow.data import Dataset
from psiflow.execution import ReferenceExecutionDefinition


@pytest.fixture
def fake_cp2k_input():
    return  """
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
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    return {
            'BASIS_SET_FILE_NAME': basis,
            'POTENTIAL_FILE_NAME': potential,
            'PARAMETER_FILE_NAME': dftd3,
            }


@pytest.fixture
def cp2k_input():
    return """
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
   &END SUBSYS
!   &PRINT
!      &STRESS_TENSOR ON
!      &END STRESS_TENSOR
!      &FORCES
!      &END FORCES
!   &END PRINT
&END FORCE_EVAL
"""


def test_reference_emt(context, dataset):
    reference = EMTReference(context)
    atoms = reference.evaluate(dataset[0])
    assert isinstance(atoms, AppFuture)
    assert isinstance(atoms.result(), FlowAtoms)
    assert atoms.result().info['evaluation_flag'] == 'success'
    assert atoms.result().evaluation_log == ''

    #evaluated = reference.evaluate(dataset)
    #evaluated.length().result()
    #assert isinstance(evaluated, Dataset)
    #assert isinstance(evaluated.data_future, DataFuture)
    #assert evaluated.length().result() == dataset.length().result()
    #for i in range(evaluated.length().result()):
    #    assert evaluated[i].result().evaluation_flag == 'success'
    #    assert evaluated[i].result().evaluation_log is None # not retained with batch eval!


def test_cp2k_insert_filepaths(fake_cp2k_input):
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
    sample = Cp2kInput.from_string(insert_filepaths_in_input(fake_cp2k_input, filepaths))
    assert str(target) == str(sample)


def test_cp2k_insert_atoms(tmp_path, fake_cp2k_input):
    atoms = FlowAtoms(numbers=np.ones(3), cell=np.eye(3), positions=np.eye(3), pbc=True)
    sample = Cp2kInput.from_string(insert_atoms_in_input(fake_cp2k_input, atoms))
    assert 'COORD' in sample['FORCE_EVAL']['SUBSYS'].subsections.keys()
    assert 'CELL' in sample['FORCE_EVAL']['SUBSYS'].subsections.keys()
    natoms = len(sample['FORCE_EVAL']['SUBSYS']['COORD'].keywords['H'])
    assert natoms == 3


def test_cp2k_success(context, cp2k_input, cp2k_data):
    reference = CP2KReference(context, cp2k_input=cp2k_input, cp2k_data=cp2k_data)
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=5 * np.eye(3),
            positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
            pbc=True,
            )
    evaluated = reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    # calculation will fail if time_per_singlepoint in execution definition is too low!
    assert evaluated.result().evaluation_flag == 'success'
    #evaluated.result()
    assert 'energy' in evaluated.result().info.keys()
    assert 'stress' in evaluated.result().info.keys()
    assert 'forces' in evaluated.result().arrays.keys()
    assert np.allclose(
            -1.165271084838365 / molmod.units.electronvolt,
            evaluated.result().info['energy'],
            )
    forces_reference = np.array([[0.01218794, 0.00001251, 0.00001251],
            [-0.01215503, 0.00001282, 0.00001282]])
    forces_reference /= molmod.units.electronvolt
    forces_reference *= molmod.units.angstrom
    assert np.allclose(forces_reference, evaluated.result().arrays['forces'])
    stress_reference = np.array([
             [4.81790309081E-01,   7.70485237955E-05,   7.70485237963E-05],
             [7.70485237955E-05,  -9.50069820373E-03,   1.61663002757E-04],
             [7.70485237963E-05,   1.61663002757E-04,  -9.50069820373E-03]])
    stress_reference *= 1000
    assert np.allclose(stress_reference, evaluated.result().info['stress'])

    # check number of mpi processes
    content = evaluated.result().evaluation_log
    ncores = context[ReferenceExecutionDefinition].ncores
    lines = content.split('\n')
    for line in lines:
        if 'Total number of message passing processes' in line:
            nprocesses = int(line.split()[-1])
        #print(line)
        if 'Number of threads for this process' in line:
            nthreads = int(line.split()[-1])
    assert nprocesses == ncores
    assert nthreads == 1 # hardcoded into app


def test_cp2k_failure(context, cp2k_data):
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
    reference = CP2KReference(context, cp2k_input=cp2k_input, cp2k_data=cp2k_data)
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=5 * np.eye(3),
            positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
            pbc=True,
            )
    evaluated = reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().evaluation_flag == 'failed'
    assert 'energy' not in evaluated.result().info.keys()
    log = evaluated.result().evaluation_log
    assert 'ABORT' in log # verify error is captured
    assert 'requested basis set' in log
    parsed = parse_evaluation_logs([evaluated.result()])
    assert 'ABORT' in parsed # verify error is captured
    assert 'requested basis set' in parsed
    assert 'INDEX 00000 - ' in parsed


def test_cp2k_timeout(context, cp2k_data, cp2k_input):
    reference = CP2KReference(context, cp2k_input=cp2k_input, cp2k_data=cp2k_data)
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=20 * np.eye(3), # box way too large
            positions=np.array([[0, 0, 0], [3, 0, 0]]),
            pbc=True,
            )
    evaluated = reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().evaluation_flag == 'failed'
    assert 'energy' not in evaluated.result().info.keys()
