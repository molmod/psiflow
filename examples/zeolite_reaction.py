from pathlib import Path

from ase.io import read

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.learning import SequentialLearning, load_learning
from psiflow.metrics import Metrics
from psiflow.models import MACEConfig, MACEModel
from psiflow.reference import CP2KReference
from psiflow.state import load_state
from psiflow.walkers import BiasedDynamicWalker, PlumedBias


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

coord1: COORDINATION GROUPA=109 GROUPB=88 R_0=1.4
coord2: COORDINATION GROUPA=109 GROUPB=53 R_0=1.4
CV: MATHEVAL ARG=coord1,coord2 FUNC=x-y PERIODIC=NO
cv2: MATHEVAL ARG=coord1,coord2 FUNC=x+y PERIODIC=NO
lwall: LOWER_WALLS ARG=cv2 AT=0.65 KAPPA=5000.0
METAD ARG=CV SIGMA=0.2 HEIGHT=5 PACE=100
"""
    return PlumedBias(plumed_input)


def get_reference():
    with open(Path.cwd() / "data" / "cp2k_input.txt", "r") as f:
        cp2k_input = f.read()
    return CP2KReference(cp2k_input=cp2k_input)


def main(path_output):
    assert not path_output.exists()
    reference = get_reference()
    bias = get_bias()

    atoms = FlowAtoms.from_atoms(read(Path.cwd() / "data" / "zeolite_proton.xyz"))
    atoms.canonical_orientation()  # transform into conventional lower-triangular box

    config = MACEConfig()
    config.r_max = 6.0
    config.num_channels = 32
    config.max_L = 1
    config.batch_size = 4
    config.patience = 10
    model = MACEModel(config)

    model.add_atomic_energy("H", reference.compute_atomic_energy("H", box_size=6))
    model.add_atomic_energy("O", reference.compute_atomic_energy("O", box_size=6))
    model.add_atomic_energy("Si", reference.compute_atomic_energy("Si", box_size=6))
    model.add_atomic_energy("Al", reference.compute_atomic_energy("Al", box_size=6))

    # set learning parameters and do pretraining
    learning = SequentialLearning(
        path_output=path_output,
        niterations=10,
        train_valid_split=0.9,
        train_from_scratch=True,
        metrics=Metrics("zeolite_reaction", "psiflow_examples"),
        error_thresholds_for_reset=(10, 200),  # in meV/atom, meV/angstrom
        temperature_ramp=(300, 1000, 5),
    )

    # construct walkers; biased MTD MD in this case
    walkers = BiasedDynamicWalker.multiply(
        50,
        data_start=Dataset([atoms]),
        bias=bias,
        timestep=0.5,
        steps=5000,
        step=50,
        start=0,
        temperature=300,
        max_excess_temperature=1000,  # reset if T > T_0 + 1000
        pressure=0,
    )
    data = learning.run(  # noqa: F841
        model=model,
        reference=reference,
        walkers=walkers,
    )


def restart(path_output):
    reference = get_reference()
    learning = load_learning(path_output)
    model, walkers, data_train, data_valid = load_state(path_output, "5")
    learning.run(
        model=model,
        reference=reference,
        walkers=walkers,
        initial_data=data_train + data_valid,
    )


if __name__ == "__main__":
    psiflow.load()
    main(Path.cwd() / "output")
    psiflow.wait()
