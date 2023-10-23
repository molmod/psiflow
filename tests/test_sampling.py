import numpy as np
import parsl

from psiflow.committee import Committee
from psiflow.data import Dataset, FlowAtoms
from psiflow.metrics import Metrics, log_dataset
from psiflow.models import MACEModel
from psiflow.reference import EMTReference
from psiflow.sampling import sample_with_committee, sample_with_model
from psiflow.walkers import BiasedDynamicWalker, DynamicWalker, PlumedBias, RandomWalker


def test_sample_metrics(mace_model, dataset, tmp_path):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    state = FlowAtoms(
        numbers=101 * np.ones(3),
        positions=np.zeros((3, 3)),
        cell=np.eye(3),
        pbc=True,
    )
    walkers.append(RandomWalker(state, seed=10))
    walkers.append(DynamicWalker(dataset[0], steps=100, step=1, start=0))
    walkers.append(DynamicWalker(dataset[0], steps=10, step=1, start=0))
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: VOLUME
CV1: MATHEVAL ARG=CV VAR=a FUNC=3*a PERIODIC=NO
restraint: RESTRAINT ARG=CV1 AT=150 KAPPA=1
"""
    bias = PlumedBias(plumed_input)
    walkers.append(BiasedDynamicWalker(dataset[0], bias=bias, seed=10, steps=10))

    reference = EMTReference()

    assert not np.allclose(dataset[0].result().arrays["forces"], 0.0)

    identifier = 4
    metrics = Metrics(wandb_group="test_sample_metrics", wandb_project="psiflow")
    data, identifier = sample_with_model(
        mace_model,
        reference,
        walkers,
        identifier,
        error_thresholds_for_reset=(1e9, 1e9),
        metrics=metrics,
    )
    assert data.length().result() == 6  # one state should have failed
    for i in range(5):
        assert data[i].result().reference_status  # should be successful
        assert "identifier" in data[i].result().info.keys()
        assert data[i].result().info["identifier"] >= 4
        assert data[i].result().info["identifier"] <= 9

    dataset_log = log_dataset(
        inputs=[data.data_future, mace_model.evaluate(data).data_future]
    )
    dataset_log = dataset_log.result()
    for key in dataset_log:
        assert len(dataset_log[key]) == 6
    assert "CV1" in dataset_log
    assert "identifier" in dataset_log
    assert 'stdout' in dataset_log
    assert sum([a is None for a in dataset_log["CV1"]]) == 6 - 1

    assert len(metrics.walker_logs) == len(walkers)
    metrics.save(tmp_path, model=mace_model, dataset=data)
    parsl.wait_for_current_tasks()
    assert (tmp_path / "walkers.log").exists()
    assert (tmp_path / "dataset.log").exists()

    data, identifier = sample_with_model(  # test without metrics
        mace_model,
        reference,
        walkers,
        identifier,
        error_thresholds_for_reset=(0, 0),
    )
    for walker in walkers:  # should all be reset
        assert walker.counter.result() == 0


def test_sample_committee(gpu, mace_config, dataset, tmp_path):
    walkers = RandomWalker.multiply(3, data_start=dataset)
    walkers.append(DynamicWalker(dataset[0], steps=100, step=1, start=0))
    walkers.append(
        DynamicWalker(dataset[0], steps=10, step=1, start=0, force_threshold=1e-7)
    )

    reference = EMTReference()
    mace_config["max_num_epochs"] = 1
    models = [MACEModel(mace_config) for i in range(4)]
    committee = Committee(models)
    committee.train(dataset[:5], dataset[5:10])

    identifier = 0
    metrics = Metrics(wandb_group="test_sample_committee", wandb_project="psiflow")
    data, identifier = sample_with_committee(
        committee,
        reference,
        walkers,
        identifier,
        nstates=3,
        error_thresholds_for_reset=(1e9, 1e9),
        metrics=metrics,
    )
    for i in range(3):
        assert data[i].result().reference_status  # should be successful
        assert "identifier" in data[i].result().info.keys()
        assert data[i].result().info["identifier"] <= 2
    assert len(metrics.walker_logs) == len(walkers)
    metrics.save(tmp_path)
    parsl.wait_for_current_tasks()
    assert (tmp_path / "walkers.log").exists()
    with open(tmp_path / "walkers.log", "r") as f:
        print(f.read())

    data, identifier = sample_with_committee(  # without metric
        committee,
        reference,
        walkers,
        identifier,
        nstates=3,
        error_thresholds_for_reset=(0, 0),
    )
    for walker in walkers:
        assert walker.counter.result() == 0  # all reset
