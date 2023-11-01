import shutil

import pytest

import psiflow
from psiflow.learning import IncrementalLearning, SequentialLearning, load_learning
from psiflow.metrics import Metrics
from psiflow.models import MACEModel
from psiflow.reference import EMTReference
from psiflow.utils import apply_temperature_ramp
from psiflow.walkers import BiasedDynamicWalker, PlumedBias, RandomWalker


def test_learning_save_load(gpu, tmp_path):
    path_output = tmp_path / "output"
    path_output.mkdir()
    SequentialLearning(
        path_output=path_output,
        pretraining_nstates=100,
    )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 100

    shutil.rmtree(path_output)
    path_output.mkdir()
    metrics = Metrics(
        wandb_project="pytest",
        wandb_group="test_learning_save_load",
    )
    SequentialLearning(
        path_output=path_output,
        metrics=metrics,
        pretraining_nstates=99,
    )
    learning_ = load_learning(path_output)
    assert learning_.pretraining_nstates == 99
    assert learning_.metrics is not None


def test_sequential_learning(gpu, tmp_path, mace_config, dataset):
    path_output = tmp_path / "output"
    path_output.mkdir()
    reference = EMTReference()
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
METAD ARG=CV SIGMA=100 HEIGHT=2 PACE=1 LABEL=metad FILE=test_hills
"""
    bias = PlumedBias(plumed_input)

    metrics = Metrics("pytest", "test_sequential_index")

    learning = SequentialLearning(
        path_output=path_output,
        metrics=metrics,
        pretraining_nstates=50,
        train_from_scratch=True,
        train_valid_split=0.8,
        niterations=1,
    )
    model = MACEModel(mace_config)
    model.config_raw["max_num_epochs"] = 1
    model.initialize(dataset[:2])

    walkers = RandomWalker.multiply(
        5,
        dataset,
        amplitude_pos=0.05,
        amplitude_box=0,
    )
    assert dataset.assign_identifiers().result() == dataset.labeled().length().result()
    data = learning.run(model, reference, walkers, dataset)
    psiflow.wait()
    assert data.labeled().length().result() == len(walkers) + dataset.length().result()
    assert learning.identifier.result() == 25
    assert data.assign_identifiers().result() == 25

    data = learning.run(model, reference, walkers)
    assert data.length().result() == 0  # iteration 0 already performed

    model.reset()
    metrics = Metrics("pytest", "test_sequential_cv")
    path_output = tmp_path / "output_"
    path_output.mkdir()
    atomic_energies = {
        "H": reference.compute_atomic_energy("H", 5),
        "Cu": reference.compute_atomic_energy("Cu", 5),
    }
    learning = SequentialLearning(
        path_output=path_output,
        metrics=metrics,
        pretraining_nstates=50,
        atomic_energies=atomic_energies,
        train_from_scratch=True,
        train_valid_split=0.8,
        niterations=1,
    )
    walkers = BiasedDynamicWalker.multiply(
        5,
        dataset,
        bias=bias,
        steps=10,
        step=1,
        temperature=300,
    )
    data = learning.run(model, reference, walkers)
    psiflow.wait()
    assert model.do_offset
    assert len(model.atomic_energies) > 0
    assert (path_output / "pretraining").is_dir()
    assert data.length().result() == 55

    # should notice that this energy is different from the one with which
    # the model was initialized
    learning.atomic_energies["H"] = 1000
    with pytest.raises(AssertionError):
        data = learning.run(model, reference, walkers)
        data.data_future.result()


def test_incremental_learning(gpu, tmp_path, mace_config, dataset):
    model = MACEModel(mace_config)
    model.initialize(dataset[:2])
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
MOVINGRESTRAINT ARG=CV STEP0=0 AT0=150 KAPPA0=1 STEP1=1000 AT1=200 KAPPA1=1
"""
    bias = PlumedBias(plumed_input)
    walkers = BiasedDynamicWalker.multiply(
        5,
        dataset,
        bias=bias,
        steps=10,
        step=1,
        temperature=300,
    )
    reference = EMTReference()
    with pytest.raises(AssertionError):
        learning = IncrementalLearning(
            tmp_path,
        )
        learning.run(
            model,
            reference,
            walkers,
        )
    learning = IncrementalLearning(
        tmp_path,
        cv_name="CV",
        cv_start=140,
        cv_stop=200,
        cv_delta=30,
        niterations=1,
        error_thresholds_for_reset=(1e9, 1e12),  # never reset
    )
    data = learning.run(
        model,
        reference,
        walkers,
    )
    assert data.length().result() == len(walkers)  # perform 1 iteration
    for walker in walkers:
        assert not walker.is_reset().result()
        steps, kappas, centers = walker.bias.get_moving_restraint(variable="CV")
        assert steps == 10
        assert centers[1] == 170  # update(initialize=True) does not change anything


def test_temperature_ramp(context):
    assert apply_temperature_ramp(100, 300, 1, 100) == 300
    assert apply_temperature_ramp(100, 500, 3, 550) == 500
    T = 100
    for _ in range(3):
        T = apply_temperature_ramp(100, 500, 5, T)
    assert T == 1 / (1 / 100 - 3 * (1 / 100 - 1 / 500) / 4)
    assert not T == 500
    T = apply_temperature_ramp(100, 500, 5, T)
    assert T == 500
