from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr, Ha
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry, NullState
from psiflow.reference import CP2K, D3, GPAW, evaluate
from psiflow.reference._cp2k import dict_to_str, parse_cp2k_output, str_to_dict


@pytest.fixture
def simple_cp2k_input():
    return """
&GLOBAL
    PRINT_LEVEL LOW
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
    18 OT LS       0.38E+00    0.1                      -14.2029934070
    19 OT CG       0.38E+00    0.2     0.00000062       -14.2029934070 -3.99E-11

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

 !-----------------------------------------------------------------------------!
                     Mulliken Population Analysis

 #  Atom  Element  Kind  Atomic population (alpha,beta) Net charge  Spin moment
       1     O        1         6.000000     0.000000     0.000000     6.000000
 # Total charge and spin        6.000000     0.000000     0.000000     6.000000

 !-----------------------------------------------------------------------------!

 !-----------------------------------------------------------------------------!
                           Hirshfeld Charges

  #Atom  Element  Kind  Ref Charge     Population       Spin moment  Net charge
      1       O      1       6.000    5.984   0.000            5.984      0.016

  Total Charge                                                            0.016
 !-----------------------------------------------------------------------------!

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

    """
    geometry = Geometry.from_data(
        numbers=np.array([8]),
        positions=np.zeros((1, 3)),
        cell=np.zeros((3, 3)),
    )
    geometry = parse_cp2k_output(cp2k_output_str, ("energy", "forces"), geometry)
    assert geometry.energy == -14.202993407031412 * Ha


def test_reference_d3(context, dataset, tmp_path):
    reference = D3(method="pbe", damping="d3bj")
    state = evaluate(dataset[-1], reference).result()
    assert state.energy is not None
    assert state.energy < 0.0  # dispersion is attractive

    subset = dataset[:3]
    data = subset.evaluate(reference)
    energy = reference.compute(subset, "energy")
    forces = reference.compute(subset, "forces")

    assert np.allclose(
        data.get("energy").result(),
        energy.result(),
    )
    assert np.allclose(
        reference.compute_atomic_energy("H").result(),
        0.0,
    )

    assert len(forces.result().shape) == 3

@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_success(context, simple_cp2k_input):
    reference = CP2K(simple_cp2k_input)
    geometry = Geometry.from_data(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        cell=5 * np.eye(3),
    )

    evaluated = evaluate(geometry, reference)
    assert isinstance(evaluated, AppFuture)

    geometry = evaluated.result()
    assert geometry != NullState

    assert Path(geometry.stdout).is_file()
    assert geometry.energy is not None
    assert not np.any(np.isnan(geometry.per_atom.forces))
    assert np.allclose(
        -1.167407360449355 * Ha,
        geometry.energy,
    )
    forces_reference = np.array([[-0.00968014, 0.0, 0.0], [0.00967947, 0.0, 0.0]])
    forces_reference *= Ha
    forces_reference /= Bohr
    assert np.allclose(
        forces_reference,
        geometry.per_atom.forces,
        atol=1e-5,
    )

    # check whether NullState evaluates to NullState
    state = evaluate(NullState, reference)
    assert state.result() == NullState

    # check number of mpi processes
    with open(geometry.stdout, "r") as f:
        content = f.read()
    definition = psiflow.context().definitions["CP2K"]
    ncores = definition.cores_per_worker
    lines = content.split("\n")
    for line in lines:
        if "Total number of message passing processes" in line:
            nprocesses = int(line.split()[-1])
        if "Number of threads for this process" in line:
            nthreads = int(line.split()[-1])
    assert ncores == nprocesses
    assert 1 == nthreads


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_failure(context, tmp_path):
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
    geometry = Geometry.from_data(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        cell=5 * np.eye(3),
    )
    evaluated = evaluate(geometry, reference)
    assert isinstance(evaluated, AppFuture)
    state = evaluated.result()
    assert state.energy is None
    assert np.all(np.isnan(state.per_atom.forces))
    with open(state.stdout, "r") as f:
        log = f.read()
    assert "ABORT" in log  # verify error is captured


def test_cp2k_memory(context, simple_cp2k_input):
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
def test_cp2k_timeout(context, simple_cp2k_input):
    reference = CP2K(simple_cp2k_input)
    geometry = Geometry.from_data(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [3, 0, 0]]),
        cell=20 * np.eye(3),  # box way too large
    )
    energy, forces = reference.compute(Dataset([geometry]))
    energy, forces = energy.result(), forces.result()
    assert np.all(np.isnan(energy))
    print(energy.shape, forces.shape)


def test_cp2k_energy(context, simple_cp2k_input):
    reference = CP2K(simple_cp2k_input, outputs=("energy",))
    geometry = Geometry.from_data(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [3, 0, 0]]),
        cell=5 * np.eye(3),  # box way too large
    )
    state = evaluate(geometry, reference).result()
    assert state.energy is not None
    assert state.stdout is not None
    assert np.all(np.isnan(state.per_atom.forces))


@pytest.mark.filterwarnings("ignore:Original input file not found")
def test_cp2k_atomic_energies(
    dataset, simple_cp2k_input
):  # use energy-only because why not
    reference = CP2K(simple_cp2k_input, outputs=("energy",), executor="CP2K_container")
    element = "H"
    energy = reference.compute_atomic_energy(element, box_size=4)
    assert abs(energy.result() - (-13.6)) < 1  # reasonably close to exact value


def test_cp2k_serialize(dataset, simple_cp2k_input):
    element = "H"
    reference = CP2K(simple_cp2k_input, outputs=("energy",))
    assert "outputs" in reference._attrs
    assert "cp2k_input_dict" in reference._attrs
    assert "cp2k_input_str" in reference._attrs
    energy = reference.compute_atomic_energy(element, box_size=4)

    data = psiflow.serialize(reference).result()
    reference = psiflow.deserialize(data)
    assert type(reference.outputs) is list
    assert np.allclose(
        energy.result(),
        reference.compute_atomic_energy(element, box_size=4).result(),
    )


def test_cp2k_posthf(context):
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
    geometry = Geometry.from_data(  # simple H2 at ~optimized interatomic distance
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        cell=5 * np.eye(3),
    )
    assert evaluate(geometry, reference).result().energy is not None


def test_gpaw_single(dataset, dataset_h2):
    gpaw = GPAW(
        mode="fd",
        nbands=0,
        xc="LDA",
        h=0.1,
        minimal_box_multiple=2,
    )
    state = evaluate(dataset_h2[0], gpaw).result()
    assert state.energy is not None
    assert state.energy < 0.0
    assert np.allclose(
        state.per_atom.positions,
        dataset_h2[0].result().per_atom.positions,
    )
    gpaw = GPAW(
        mode="fd",
        nbands=0,
        xc="LDA",
        h=0.1,
        minimal_box_multiple=2,
        executor="GPAW_container",
    )
    energy = gpaw.compute(dataset_h2[:1])[0].result()
    assert np.allclose(state.energy, energy)
    assert gpaw.compute_atomic_energy("Zr", box_size=9).result() == 0.0

    gpaw = GPAW(askdfj="asdfk")  # invalid input
    assert evaluate(dataset_h2[1], gpaw).result().energy is None
