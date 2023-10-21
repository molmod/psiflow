import requests
import logging
from pathlib import Path
import numpy as np

from ase.build import bulk, make_supercell

import psiflow
from psiflow.learning import SequentialLearning, load_learning, CommitteeLearning
from psiflow.models import MACEModel, MACEConfig
from psiflow.reference import EMTReference
from psiflow.data import FlowAtoms, Dataset
from psiflow.walkers import DynamicWalker, PlumedBias
from psiflow.state import load_state
from psiflow.metrics import Metrics
from psiflow.committee import Committee


def main(path_output):
    path_sequential = path_output / "learn_sequential"
    path_sequential.mkdir(parents=True)
    path_committee = path_output / "learn_committee"
    path_committee.mkdir(parents=True)

    reference = EMTReference()  # CP2K; PBE-D3(BJ); TZVP
    atoms = make_supercell(bulk("Cu", "fcc", a=3.6, cubic=True), 3 * np.eye(3))

    config = MACEConfig()
    config.r_max = 6.0
    config.num_channels = 4
    config.max_L = 1
    config.batch_size = 4
    config.patience = 4
    model = MACEModel(config)

    model.add_atomic_energy("Cu", reference.compute_atomic_energy("Cu", box_size=6))

    # set learning parameters
    learning = SequentialLearning(
        path_output=path_sequential,
        niterations=1,
        pretraining_nstates=50,
        train_valid_split=0.9,
        train_from_scratch=True,
        metrics=Metrics("copper_EMT", "psiflow_examples"),
        error_thresholds_for_reset=(10, 200),  # in meV/atom, meV/angstrom
        temperature_ramp=(400, 2000, 1),
    )

    # construct walkers; straightforward MD in this case
    walkers = DynamicWalker.multiply(
        5,
        data_start=Dataset([atoms]),
        timestep=0.5,
        steps=200,
        step=40,
        start=0,
        temperature=100,
        temperature_threshold=300,  # reset if T > T_0 + 300 K
        pressure=0,
    )
    data = learning.run(
        model=model,
        reference=reference,
        walkers=walkers,
    )
    model.reset()

    # continue with committee learning with modified temperature
    for walker in walkers:
        walker.temperature = 2000
    learning = CommitteeLearning(
        path_committee,
        niterations=3,
        metrics=Metrics("copper_EMT", "psiflow_examples"),
        error_thresholds_for_reset=(10, 200),  # in meV/atom, meV/angstrom
        temperature_ramp=None,
        nstates_per_iteration=3,
    )
    committee = Committee([model.copy() for i in range(2)])
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
