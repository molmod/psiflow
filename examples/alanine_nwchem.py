from pathlib import Path

from ase.io import read

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.reference import NWChemReference
from psiflow.walkers import PlumedBias, BiasedDynamicWalker, DynamicWalker
from psiflow.models import NequIPModel, NequIPConfig
from psiflow.metrics import Metrics
from psiflow.learning import SequentialLearning


def get_reference():
    calculator_kwargs = {
        "basis": {e: "3-21g" for e in ["H", "C", "O", "N"]},
        "dft": {
            "xc": "pw91lda",
            "mult": 1,
            "convergence": {
                "energy": 1e-6,
                "density": 1e-6,
                "gradient": 1e-6,
            },
            #'disp': {'vdw': 3},
        },
    }
    return NWChemReference(**calculator_kwargs)


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17

METAD  LABEL=bias ARG=phi,psi SIGMA=0.2,0.2 GRID_MIN=-pi,-pi GRID_MAX=pi,pi HEIGHT=0.41 PACE=5 TEMP=300 BIASFACTOR=10
"""
    return PlumedBias(plumed_input)


def main(path_output):
    assert not path_output.exists()
    atoms = FlowAtoms.from_atoms(read("data/alanine.xyz"))
    reference = get_reference()
    bias = get_bias()

    config = NequIPConfig()
    config.r_max = 4.0
    config.num_features = 16
    config.invariant_layers = 1
    config.invariant_neurons = 32
    config.batch_size = 4
    config.loss_coeffs["total_energy"][0] = 50
    config.early_stopping_patiences["validation_loss"] = 16
    model = NequIPModel(config)

    model.add_atomic_energy("H", reference.compute_atomic_energy("H"))
    model.add_atomic_energy("C", reference.compute_atomic_energy("C"))
    model.add_atomic_energy("O", reference.compute_atomic_energy("O"))
    model.add_atomic_energy("N", reference.compute_atomic_energy("N"))

    learning = SequentialLearning(
        path_output,
        niterations=10,
        pretraining_nstates=50,
        pretraining_amplitude_pos=0.05,
        metrics=Metrics("alanine_nwchem", "psiflow_examples"),
        error_thresholds_for_reset=(5, 100),
        temperature_ramp=(100, 1000, 5),
    )

    # construct walkers; mix metadynamics with regular MD
    walkers_md = DynamicWalker.multiply(
        20,
        data_start=Dataset([atoms]),
        timestep=0.5,
        steps=20,
        step=1,
        temperature=300,
        max_excess_temperature=300,  # reset if T > T0 + 200
    )
    walkers_mtd = BiasedDynamicWalker.multiply(
        20,
        data_start=Dataset([atoms]),
        bias=bias,
        timestep=0.5,
        steps=30,
        step=1,
        temperature=300,
        max_excess_temperature=300,  # reset if T > T0 + 200
    )

    # this is mostly a toy script to test code paths rather
    # than an actually working example
    data = learning.run(
        model=model,
        reference=reference,
        walkers=walkers_md + walkers_mtd,
    )


if __name__ == "__main__":
    psiflow.load()
    path_output = Path.cwd() / "output"
    main(path_output)
