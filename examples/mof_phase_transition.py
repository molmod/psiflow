from pathlib import Path

from ase.io import read

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.learning import IncrementalLearning, load_learning
from psiflow.metrics import Metrics
from psiflow.models import MACEConfig, MACEModel
from psiflow.reference import CP2KReference
from psiflow.state import load_state
from psiflow.walkers import BiasedDynamicWalker, PlumedBias


def get_bias():
    """Defines the metadynamics parameters based on a plumed input script"""
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
MOVINGRESTRAINT ARG=CV STEP0=0 AT0=5250 KAPPA0=0.1 STEP1=5000 AT1=5000 KAPPA1=0.1
"""
    return PlumedBias(plumed_input)


def get_reference():
    with open(Path.cwd() / "data" / "cp2k_input.txt", "r") as f:
        cp2k_input = f.read()
    return CP2KReference(cp2k_input_str=cp2k_input)


def main(path_output):
    assert not path_output.exists()
    reference = get_reference()  # CP2K; PBE-D3(BJ); TZVP

    atoms = FlowAtoms.from_atoms(read(Path.cwd() / "data" / "mof.xyz"))
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
    model.add_atomic_energy("C", reference.compute_atomic_energy("C", box_size=6))
    model.add_atomic_energy("Al", reference.compute_atomic_energy("Al", box_size=6))

    # set learning parameters
    learning = IncrementalLearning(
        path_output=path_output,
        niterations=10,
        train_valid_split=0.9,
        train_from_scratch=True,
        metrics=Metrics("MOF_phase_transition", "psiflow_examples"),
        error_thresholds_for_reset=(10, 200),  # in meV/atom, meV/angstrom
        cv_name="CV",
        cv_start=5250,
        cv_stop=3000,
        cv_delta=-250,
    )

    bias = get_bias()
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
    path_output = Path.cwd() / "output"  # stores learning results
    main(path_output)
    psiflow.wait()
