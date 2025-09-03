import numpy as np
import pytest
from ase.units import Bohr, Ha
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, NullState
from psiflow.reference import CP2K, GPAW, ORCA, create_orca_input
from psiflow.reference.reference import Status
from psiflow.reference.cp2k_ import (
    dict_to_str,
    str_to_dict,
    parse_output as parse_cp2k_output,
)
from psiflow.utils.parse import get_task_logs


@pytest.fixture
def simple_cp2k_input() -> str:
    return """
&GLOBAL
    PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
   METHOD Quickstep
   &PRINT
    &FORCES
    &END FORCES
   &END PRINT
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  BASIS_MOLOPT_UZH
      BASIS_SET_FILE_NAME  BASIS_ADMM_MOLOPT
      POTENTIAL_FILE_NAME  POTENTIAL_UZH
      &XC
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
         &VDW_POTENTIAL
            &PAIR_POTENTIAL
               PARAMETER_FILE_NAME  dftd3.dat
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &COORD
        H 0 0 0
        F 1 0 0
        C 2 0 0
        H 3 0 0
        C -1 0 0
      &END COORD
      &KIND H
         ELEMENT  H
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q1
         POTENTIAL GTH-PBE-q1
      &END KIND
      &KIND F
         ELEMENT  F
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q7
         POTENTIAL GTH-PBE-q7
      &END KIND
    &END SUBSYS
&END FORCE_EVAL
"""


@pytest.fixture
def geom_h2_p() -> Geometry:
    # periodic H2 at ~optimized interatomic distance
    pos = np.array([[0, 0, 0], [0.74, 0, 0]])
    return Geometry.from_data(numbers=np.ones(2), positions=pos, cell=5 * np.eye(3))


def test_cp2k_check_input(simple_cp2k_input):
    cp2k_input = str_to_dict(simple_cp2k_input)
    assert cp2k_input["force_eval"]["method"] == "quickstep"
    assert "BASIS_MOLOPT_UZH" in cp2k_input["force_eval"]["dft"]["basis_set_file_name"]
    assert "BASIS_ADMM_MOLOPT" in cp2k_input["force_eval"]["dft"]["basis_set_file_name"]
    assert len(cp2k_input["force_eval"]["subsys"]["kind"]) == 2
    assert len(cp2k_input["force_eval"]["subsys"]["coord"]["*"]) == 5
    assert cp2k_input["force_eval"]["subsys"]["coord"]["*"][0].startswith("H")
    assert cp2k_input["force_eval"]["subsys"]["coord"]["*"][1].startswith("F")
    assert cp2k_input["force_eval"]["subsys"]["coord"]["*"][2].startswith("C")
    assert cp2k_input["force_eval"]["subsys"]["coord"]["*"][3].startswith("H")
    assert cp2k_input["force_eval"]["subsys"]["coord"]["*"][4].startswith("C")
    dict_to_str(cp2k_input)


def test_cp2k_parse_output():
    cp2k_output_str = """
### CP2K OUTPUT SNIPPETS ###
    
     TOTAL NUMBERS AND MAXIMUM NUMBERS

      Total number of            - Atomic kinds:                                   1
                                 - Atoms:                                          1
      
### SKIPPED A BIT ###
          
     MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM

   Atom Kind Element         X             Y             Z       Z(eff)     Mass
      1    1 O     8      0.000000      0.000000      0.000000   6.0000  15.9994

 SCF PARAMETERS         Density guess:                                   RESTART
                        --------------------------------------------------------
                        max_scf:                                              20
                        max_scf_history:                                       0
                        max_diis:                                              4
                        --------------------------------------------------------
                        eps_scf:                                        1.00E-06
                        eps_scf_history:                                0.00E+00
                        eps_diis:                                       1.00E-01
                        eps_eigval:                                     1.00E-05

### SKIPPED A BIT ###

  *** SCF run converged in    19 steps ***


  Electronic density on regular grids:         -5.9999999998        0.0000000002
  Core density on regular grids:                5.9999999999       -0.0000000001
  Total charge density on r-space grids:        0.0000000000
  Total charge density g-space grids:           0.0000000000

  Overlap energy of the core charge distribution:               0.00000000000000
  Self energy of the core charge distribution:                -41.54166754366195
  Core Hamiltonian energy:                                     11.97774883943202
  Hartree energy:                                              18.63665627699511
  Exchange-correlation energy:                                 -3.27569837713826
  Dispersion energy:                                           -0.00003260261466

  Total energy:                                               -14.20299340698774

  outer SCF iter =    4 RMS gradient =   0.62E-06 energy =        -14.2029934070
  outer SCF loop converged in   4 iterations or   79 steps


  Integrated absolute spin density  :                               5.9999999997
  Ideal and single determinant S**2 :                   12.000000      12.000000
  
### SKIPPED A BIT ###

 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:              -14.202993407031412

 ATOMIC FORCES in [a.u.]

 # Atom   Kind   Element          X              Y              Z
      1      1      O           0.00000000     0.00000000     0.00000000
 SUM OF ATOMIC FORCES           0.00000000     0.00000000     0.00000000     0.00000000

 STRESS| Analytical stress tensor [GPa]
 STRESS|                        x                   y                   z
 STRESS|      x        1.37382515001E-01   3.90453773469E-02  -3.90586881497E-02
 STRESS|      y        3.90453773469E-02   1.37453715622E-01   3.90638959674E-02
 STRESS|      z       -3.90586881497E-02   3.90638959674E-02   1.37635963266E-01
 STRESS| 1/3 Trace                                             1.37490731296E-01
 STRESS| Determinant                                           1.85075909182E-03

 STRESS| Eigenvectors and eigenvalues of the analytical stress tensor [GPa]
 STRESS|                        1                   2                   3
 STRESS| Eigenvalues   5.93786734955E-02   1.76460235520E-01   1.76633284873E-01
 STRESS|      x           0.577844347143      0.756273914341     -0.306831675291
 STRESS|      y          -0.577518702860      0.644541601612      0.501037195863
 STRESS|      z           0.576687140764     -0.112320480227      0.809207051007
 
### SKIPPED A BIT ###
 
  -------------------------------------------------------------------------------
 -                                                                             -
 -                                T I M I N G                                  -
 -                                                                             -
 -------------------------------------------------------------------------------
 SUBROUTINE                       CALLS  ASD         SELF TIME        TOTAL TIME
                                MAXIMUM       AVERAGE  MAXIMUM  AVERAGE  MAXIMUM
 CP2K                                 1  1.0    0.036    0.042  520.335  520.336
 qs_forces                            1  2.0    0.001    0.001  519.254  519.255
 qs_energies                          1  3.0    0.005    0.006  507.372  507.376
 
### SKIPPED A BIT ###

  **** **** ******  **  PROGRAM ENDED AT                 2025-07-31 13:45:43.191
 ***** ** ***  *** **   PROGRAM RAN ON                         node3594.doduo.os
 **    ****   ******    PROGRAM RAN BY                                 <unknown>
 ***** **    ** ** **   PROGRAM PROCESS ID                               1183304
  **** **  *******  **  PROGRAM STOPPED IN              /tmp/mytmpdir.2RsSPreAda

    """
    data_out = parse_cp2k_output(cp2k_output_str, ("energy", "forces"))
    assert data_out["energy"] == -14.202993407031412 * Ha
    assert data_out["runtime"] == 520.336
    assert data_out["status"] == Status.SUCCESS


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_success(simple_cp2k_input, geom_h2_p):
    reference = CP2K(simple_cp2k_input)

    future = reference.evaluate(geom_h2_p)
    future_null = reference.evaluate(NullState)
    assert isinstance(future, AppFuture)
    geom_out = future.result()
    geom_null = future_null.result()

    ref_energy = -1.167407360449355 * Ha
    ref_forces = np.array([[-0.00968014, 0.0, 0.0], [0.00967947, 0.0, 0.0]]) * Ha / Bohr
    assert geom_out != NullState
    assert geom_out.energy is not None
    assert not np.any(np.isnan(geom_out.per_atom.forces))
    assert np.allclose(ref_energy, geom_out.energy)
    assert np.allclose(ref_forces, geom_out.per_atom.forces, atol=1e-5)
    assert geom_null == NullState  # check whether NullState evaluates to NullState

    # check number of mpi processes
    stdout, _ = get_task_logs(geom_out.order["task_id"])
    lines = stdout.read_text().split("\n")
    for line in lines:
        if "Total number of message passing processes" in line:
            nprocesses = int(line.split()[-1])
        if "Number of threads for this process" in line:
            nthreads = int(line.split()[-1])
    definition = psiflow.context().definitions["CP2K"]
    ncores = definition.cores_per_worker
    assert ncores == nprocesses
    assert 1 == nthreads


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_failure(geom_h2_p):
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
    reference = CP2K(cp2k_input)
    future = reference.evaluate(geom_h2_p)
    assert isinstance(future, AppFuture)
    geom_out = future.result()
    assert geom_out.energy is None
    assert np.all(np.isnan(geom_out.per_atom.forces))
    stdout, _ = get_task_logs(geom_out.order["task_id"])
    assert "ABORT" in stdout.read_text()  # verify error is captured


def test_cp2k_memory(simple_cp2k_input):
    # TODO: test_cp2k_memory == test_cp2k_timeout until memory constraints work
    reference = CP2K(simple_cp2k_input)
    geometry = Geometry.from_data(
        numbers=np.ones(4000),
        positions=np.random.uniform(0, 20, size=(4000, 3)),
        cell=20 * np.eye(3),  # box way too large
    )
    energy, forces = reference.compute(geometry)
    energy, forces = energy.result(), forces.result()
    assert np.all(np.isnan(energy))


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_timeout(simple_cp2k_input, geom_h2_p):
    reference = CP2K(simple_cp2k_input)
    geom_h2_p.cell = 20 * np.eye(3)  # box way too large
    energy, forces = reference.compute(Dataset([geom_h2_p]))
    energy, forces = energy.result(), forces.result()
    assert np.all(np.isnan(energy))


def test_cp2k_energy(simple_cp2k_input, geom_h2_p):
    reference = CP2K(simple_cp2k_input, outputs=("energy",))
    geom_out = reference.evaluate(geom_h2_p).result()
    assert geom_out.energy is not None
    assert np.all(np.isnan(geom_out.per_atom.forces))


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_atomic_energies(simple_cp2k_input):
    # use energy-only because why not
    reference = CP2K(simple_cp2k_input, outputs=("energy",))
    energy = reference.compute_atomic_energy("H", box_size=4)
    assert abs(energy.result() - (-13.6)) < 1  # reasonably close to exact value


def test_cp2k_serialize(simple_cp2k_input):
    element = "H"
    reference = CP2K(simple_cp2k_input, outputs=("energy",))
    assert "outputs" in reference._attrs
    assert "input_dict" in reference._attrs

    data = psiflow.serialize(reference).result()
    reference2 = psiflow.deserialize(data)
    future = reference.compute_atomic_energy(element, box_size=4)
    future2 = reference2.compute_atomic_energy(element, box_size=4)

    assert type(reference2.outputs) is list
    assert np.allclose(future.result(), future2.result())


def test_cp2k_posthf(geom_h2_p):
    cp2k_input_str = """
&FORCE_EVAL
   METHOD Quickstep
   &DFT
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  BASIS_RI_cc-TZ
      POTENTIAL_FILE_NAME  POTENTIAL_UZH
      &XC
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
            &WF_CORRELATION
                &RI_RPA
                    MINIMAX_QUADRATURE
                    RPA_NUM_QUAD_POINTS 2
                    &END RI_RPA
                &END WF_CORRELATION
      &END XC
   &END DFT
   &SUBSYS
      &KIND H
         ELEMENT  H
            BASIS_SET cc-DZ
            POTENTIAL GTH-PBE-q1
      &END KIND
    &END SUBSYS
&END FORCE_EVAL
"""
    reference = CP2K(cp2k_input_str, outputs=("energy",))
    assert reference.evaluate(geom_h2_p).result().energy is not None


def test_gpaw_single(dataset_h2):
    parameters = dict(mode="fd", nbands=0, xc="LDA", h=0.3, minimal_box_multiple=2)
    gpaw = GPAW(parameters)
    future_in = dataset_h2[0]
    future_out = gpaw.evaluate(future_in)
    future_energy = gpaw.compute(dataset_h2[:1])[0]
    future_energy_zr = gpaw.compute_atomic_energy("Zr", box_size=9)
    gpaw = GPAW({"askdfj": "asdfk"})  # invalid input
    future_fail = gpaw.evaluate(future_in)

    geom_in, geom_out = future_in.result(), future_out.result()
    energy, energy_zr = future_energy.result(), future_energy_zr.result()

    assert geom_out.energy is not None and geom_out.energy < 0.0
    assert np.allclose(geom_out.per_atom.positions, geom_in.per_atom.positions)
    assert np.allclose(geom_out.energy, energy)
    assert energy_zr == 0.0
    assert future_fail.result().energy is None


# TODO: enable once we have an ORCA container
# def test_orca_single(dataset_h2):
#     input_str = create_orca_input()
#     orca = ORCA(input_str)
#     future_in = dataset_h2[0]
#     future_out = orca.evaluate(future_in)
#     future_energy = orca.compute(dataset_h2[:1])[0]
#     future_energy_h = orca.compute_atomic_energy("H")
#
#     geom_in, geom_out = future_in.result(), future_out.result()
#     energy, energy_h = future_energy.result(), future_energy_h.result()
#
#     assert geom_out.energy is not None and geom_out.energy < 0.0
#     assert np.allclose(geom_out.per_atom.positions, geom_in.per_atom.positions)
#     assert np.allclose(geom_out.energy, energy)
#     print(energy, energy_h)
#     assert np.isclose(energy_h, -13.6)
