from pathlib import Path

import requests
from ase.io import read

import psiflow
from psiflow.committee import Committee
from psiflow.data import Dataset, FlowAtoms
from psiflow.learning import CommitteeLearning, SequentialLearning
from psiflow.metrics import Metrics
from psiflow.models import MACEConfig, MACEModel
from psiflow.reference import CP2KReference
from psiflow.walkers import BiasedDynamicWalker, PlumedBias


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

c1: COM ATOMS=60,44
c2: COM ATOMS=60,56

d1: DISTANCE ATOMS=c1,108
d2: DISTANCE ATOMS=c2,108

CV: MATHEVAL ARG=d1,d2 FUNC=(x-y) PERIODIC=NO

METAD ARG=CV SIGMA=0.5 HEIGHT=2 PACE=100 LABEL=metad FILE=HILLS
"""
    return PlumedBias(plumed_input)


def get_reference():
    """Defines a generic PBE-D3/TZVP reference level of theory

    Basis set, pseudopotentials, and D3 correction parameters are obtained from
    the official CP2K repository, v2023.1, and saved in the internal directory of
    psiflow. The input file is assumed to be available locally.

    """
    with open(Path.cwd() / "data" / "cp2k_input.txt", "r") as f:
        cp2k_input = f.read()
    # set UKS to True and add CHARGE 1
    cp2k_input = cp2k_input.split("\n")
    cp2k_input.insert(5, "      CHARGE 1")
    cp2k_input = "\n".join(cp2k_input)
    reference = CP2KReference(cp2k_input=cp2k_input)
    basis = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/BASIS_MOLOPT_UZH"
    ).text
    dftd3 = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/dftd3.dat"
    ).text
    potential = requests.get(
        "https://raw.githubusercontent.com/cp2k/cp2k/v2023.1/data/POTENTIAL_UZH"
    ).text
    cp2k_data = {
        "basis_set": basis,
        "potential": potential,
        "dftd3": dftd3,
    }
    for key, value in cp2k_data.items():
        with open(psiflow.context().path / key, "w") as f:
            f.write(value)
        reference.add_file(key, psiflow.context().path / key)
    return reference


def main(path_output):
    assert not path_output.exists()
    reference = get_reference()  # CP2K; PBE-D3(BJ); TZVP
    bias = get_bias()  # simple MTD bias on unit cell volume
    atoms = FlowAtoms.from_atoms(read(Path.cwd() / "data" / "perovskite_defect.xyz"))
    atoms.canonical_orientation()  # transform into conventional lower-triangular box

    config = MACEConfig()
    config.r_max = 7.0
    config.num_channels = 16
    config.max_L = 1
    config.batch_size = 4
    config.patience = 10
    config.energy_weight = 100
    model = MACEModel(config)

    model.add_atomic_energy("H", reference.compute_atomic_energy("H", box_size=6))
    model.add_atomic_energy("C", reference.compute_atomic_energy("C", box_size=6))
    model.add_atomic_energy("N", reference.compute_atomic_energy("N", box_size=6))
    model.add_atomic_energy("I", reference.compute_atomic_energy("I", box_size=6))
    model.add_atomic_energy("Pb", reference.compute_atomic_energy("Pb", box_size=6))

    walkers = BiasedDynamicWalker.multiply(
        100,
        data_start=Dataset([atoms]),
        bias=bias,
        timestep=0.5,
        steps=1000,
        step=50,
        start=0,
        temperature=100,
        max_excess_temperature=300,  # reset if T > T_0 + 300 K
        pressure=0,
    )
    metrics = Metrics("perovskite_defect", "psiflow_examples")

    learning = SequentialLearning(
        path_output=path_output / "learn_sequential",
        niterations=2,
        train_valid_split=0.9,
        metrics=metrics,
        error_thresholds_for_reset=(10, 100),  # in meV/atom, meV/angstrom
        temperature_ramp=(200, 400, 1),
    )
    data = learning.run(
        model=model,
        reference=reference,
        walkers=walkers,
    )

    # continue with committee learning
    learning = CommitteeLearning(
        path_output=path_output / "learn_committee",
        niterations=5,
        train_valid_split=0.9,
        metrics=metrics,
        error_thresholds_for_reset=(10, 100),  # in meV/atom, meV/angstrom
        temperature_ramp=(600, 1200, 3),
        nstates_per_iteration=50,
    )
    model.reset()
    committee = Committee([model.copy() for i in range(4)])
    data = learning.run(
        committee=committee,
        reference=reference,
        walkers=walkers,
        initial_data=data,
    )


if __name__ == "__main__":
    psiflow.load()
    path_output = Path.cwd() / "output"  # stores learning results
    main(path_output)
