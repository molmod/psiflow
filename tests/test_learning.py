import numpy as np
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import new_nullstate
from psiflow.hamiltonians import EinsteinCrystal
from psiflow.learning import Learning, evaluate_outputs
from psiflow.metrics import Metrics, _create_table, parse_walker_log, reconstruct_dtypes
from psiflow.reference import D3
from psiflow.sampling import SimulationOutput, Walker
from psiflow.utils.apps import combine_futures
from psiflow.utils.io import _load_metrics, _save_metrics, load_metrics, save_metrics


def test_load_save_metrics(tmp_path):
    dtypes = [
        ("a", np.float_, (2,)),
        ("b", np.bool_),
        ("c", np.int_),
        ("d", np.unicode_, 4),
    ]
    dtype = np.dtype(dtypes)
    data = np.recarray(2, dtype=np.dtype(dtypes))
    data.d[0] = "asdf"
    data.d[1] = "f"
    data.b[0] = True
    data.a[0, :] = np.array([0.2, 0.3])
    data.a[1, :] = np.array([0.0, 0.0])
    data.c[0] = 1002

    reconstructed = reconstruct_dtypes(dtype)
    for a, b in zip(dtypes, reconstructed):
        assert a == b

    path = tmp_path / "test.numpy"
    _save_metrics(
        data,
        outputs=[File(str(path))],
    )
    assert path.exists()
    data = _load_metrics(inputs=[File(str(path))])
    assert data.d[0] == "asdf"
    assert data.d[1] == "f"
    assert data.b[0]
    assert not data.b[1]
    assert np.allclose(data.a[0], np.array([0.2, 0.3]))
    assert np.allclose(data.a[1], 0.0)
    assert data.c[0] == 1002
    assert data.dtype.names == ("a", "b", "c", "d")


def test_metrics(dataset_h2):
    data = dataset_h2[:10]
    data.assign_identifiers(8)
    states = [data[i] for i in range(data.length().result())]
    states[4].result().energy = None
    states[3].result().identifier = None
    states[8].result().phase = "asldfkjasldfkjsadflkj"
    states[4].result().delta = 1.0
    states[3].result().order["test"] = 4.0
    states[4].result().order["test"] = 3.0
    states[5].result().order["a"] = 1.0
    states[7] = new_nullstate()
    states[1].result().logprob = np.array([-1.0, 1.0])

    errors = [np.random.uniform(0, 1, size=2) for i in range(len(states))]
    statuses = [0] * 10
    temperatures = [400] * 10
    times = [3.0] * 10
    resets = [False] * 10
    resets[7] = True

    data = parse_walker_log(
        combine_futures(inputs=statuses),
        combine_futures(inputs=temperatures),
        combine_futures(inputs=times),
        combine_futures(inputs=errors),
        combine_futures(inputs=states),
        combine_futures(inputs=resets),
    ).result()
    assert len(data) == 10
    assert np.allclose(data.walker_index, np.arange(10))
    assert data.phase[8] == states[8].result().phase
    assert data.identifier[3] == -1
    assert data.identifier[4] == 8 + 4  # assigned before 3rd state set to none
    assert data.delta[4] == 1.0
    assert np.allclose(data.logprob[1], np.array([-1.0, 1.0]))
    assert np.allclose(data.test[3], 4.0)
    assert np.allclose(data.test[4], 3.0)
    assert np.allclose(data.a[5], 1.0)

    s = _create_table(data)
    assert "asldfkjasldfkjsadflkj" in s

    # convert to outputs
    outputs = []
    for i in range(10):
        output = SimulationOutput([])
        output.status = statuses[i]
        output.temperature = temperatures[i]
        output.state = states[i]
        output.time = times[i]
        outputs.append(output)

    metrics = Metrics()

    metrics.log_walkers(
        outputs,
        errors,
        states,
        resets,
    )
    data = load_metrics(inputs=[metrics.metrics]).result()
    assert np.allclose(data.identifier, np.sort(data.identifier))
    assert len(data) == 10 - 2  # 2 states either NullState or with None / -1 identifier
    assert np.all(np.isnan(data.e_rmse[:, 1]))

    einstein = EinsteinCrystal(dataset_h2[3], force_constant=1)
    labeled = Dataset(states).evaluate(einstein).filter("identifier")
    einstein = EinsteinCrystal(dataset_h2[4], force_constant=1)  # different
    metrics.update(labeled, einstein)
    data_ = load_metrics(inputs=[metrics.metrics]).result()
    e_rmse = data_.e_rmse
    assert np.allclose(data.e_rmse[data.identifier != -1][:, 0], e_rmse[:, 1])
    assert np.all(~np.isnan(e_rmse[:, 0]))

    # do it twice
    metrics.log_walkers(
        outputs,
        errors,
        states,
        resets,
    )
    double_identifier = np.concatenate(
        (data.identifier, data.identifier),
        axis=0,
        dtype=np.int_,
    )
    data = load_metrics(inputs=[metrics.metrics]).result()
    assert np.allclose(data.identifier, double_identifier)
    # walker indices 3 and 7 should not end up in data
    walker_indices = np.concatenate((np.array([0, 1, 2, 4, 5, 6, 8, 9]),) * 2)
    assert np.allclose(
        data.walker_index,
        walker_indices,
    )


def test_evaluate_outputs(dataset):
    einstein = EinsteinCrystal(dataset[0], force_constant=2)
    data = dataset.reset()
    outputs = [SimulationOutput(["some", "fields"]) for i in range(10)]
    for i, output in enumerate(outputs):
        output.state = data[i]
        output.status = 0

    outputs[3].state = new_nullstate()
    outputs[7].status = 2  # should be null state

    identifier = 3
    identifier, data, resets = evaluate_outputs(
        outputs,
        einstein,
        D3(method="pbe", damping="d3bj"),
        identifier=identifier,
        error_thresholds_for_reset=[None, None],  # never reset
        error_thresholds_for_discard=[None, None],
        metrics=Metrics(),
    )
    assert identifier.result() == 3 + len(outputs) - 2
    assert data.length().result() == len(outputs) - 2
    assert data.filter("forces").length().result() == len(outputs) - 2

    assert not all([r.result() for r in resets])

    identifier, data, resets = evaluate_outputs(
        outputs,
        einstein,
        D3(method="pbe", damping="d3bj"),
        identifier=identifier,
        error_thresholds_for_reset=[0.0, 0.0],
        error_thresholds_for_discard=[0.0, 0.0],
        metrics=Metrics(),
    )
    assert all([r.result() for r in resets])


def test_wandb():
    dtypes = [
        ("e_rmse", np.float_, (2,)),
        ("f_rmse", np.float_, (2,)),
        ("reset", np.bool_),
        ("identifier", np.int_),
        ("phase", np.unicode_, 8),
        ("some_cv", np.float_),
    ]
    data = np.recarray(4, dtype=np.dtype(dtypes))

    data.identifier[:] = np.arange(4)
    data.some_cv[:] = np.arange(4)
    data.some_cv[2] = np.nan
    data.phase[1] = "asdfa"
    data.reset[1] = True

    data.e_rmse[:] = np.random.uniform(0, 2, size=(4, 2))
    data.f_rmse[:] = np.random.uniform(0, 2, size=(4, 2))
    metrics_future = save_metrics(
        data,
        outputs=[psiflow.context().new_file("metrics_", ".numpy")],
    ).outputs[0]

    metrics = Metrics("test_group", "test_project", metrics_future)
    metrics.to_wandb()

    serialized = psiflow.serialize(metrics).result()
    metrics = psiflow.deserialize(serialized)

    data_ = load_metrics(inputs=[metrics.metrics]).result()
    assert np.allclose(data.e_rmse, data_.e_rmse)
    assert np.allclose(data.some_cv, data_.some_cv, equal_nan=True)
    psiflow.wait()


def test_learning_workflow(tmp_path, gpu, mace_model, dataset):
    learning = Learning(
        D3(method="pbe", damping="d3bj"),
        tmp_path / "output",
        error_thresholds_for_reset=[None, None],
        error_thresholds_for_discard=[None, None],
    )
    assert "reference" in learning._serial
    assert "metrics" in learning._serial
    assert learning.iteration == -1
    assert learning.identifier == 0
    assert not learning.skip("-1_passive_learning")

    data = psiflow.serialize(learning).result()
    learning_ = psiflow.deserialize(data)
    learning.update(learning_)

    walkers = [
        Walker(dataset[0], EinsteinCrystal(dataset[1], 1.0)),
        Walker(dataset[0], EinsteinCrystal(dataset[5], 1.0)),
        Walker(dataset[1], EinsteinCrystal(dataset[2], 1000)),
    ]
    mace_model, walkers = learning.active_learning(
        mace_model,
        walkers,
        steps=5,
        max_force=10,
    )
    # assumes no resets happen because of error thresholds!
    assert learning.iteration == 0
    assert not walkers[0].is_reset().result()
    assert walkers[2].is_reset().result()
    psiflow.wait()

    metrics = load_metrics(inputs=[learning.metrics.metrics]).result()
    assert len(metrics) == 2  # NullStates for status not in [0, 1]
    assert np.allclose(
        metrics.identifier,
        np.array([0, 1], dtype=np.int_),
    )
    assert np.allclose(
        metrics.walker_index,
        np.array([0, 1], dtype=np.int_),
    )
    assert not np.any(np.isnan(metrics.e_rmse[:, 1]))  # should have been updated
    assert not np.any(np.isnan(metrics.f_rmse[:, 1]))

    assert learning.skip("0_active_learning")
    model_, walkers_ = learning.load("0_active_learning")
    for w, w_ in zip(walkers, walkers_):
        assert (
            w.state.result() == w_.state
        )  # no result call necessary after deserialize
        assert np.allclose(
            w.hamiltonian.compute(w.state, "energy").result(),
            w_.hamiltonian.compute(w_.state, "energy").result(),
        )

    mace_model, walkers = learning.active_learning(
        model_,
        walkers_,
        steps=5,
        max_force=10,
    )
    metrics = load_metrics(inputs=[learning.metrics.metrics]).result()
    assert len(metrics) == 4
    assert np.allclose(
        metrics.identifier,
        np.array([0, 1, 2, 3], dtype=np.int_),
    )
