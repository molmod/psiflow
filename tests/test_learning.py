import numpy as np
from parsl.data_provider.files import File

from psiflow.data import Dataset
from psiflow.geometry import new_nullstate
from psiflow.hamiltonians import EinsteinCrystal
from psiflow.learning import evaluate_outputs
from psiflow.metrics import Metrics, _create_table, parse_walker_log, reconstruct_dtypes
from psiflow.reference import EMT
from psiflow.sampling import SimulationOutput
from psiflow.utils import _load_metrics, _save_metrics, combine_futures, load_metrics


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

    errors = [tuple(np.random.uniform(0, 1, size=2)) for i in range(len(states))]
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
    labeled = einstein.evaluate(Dataset(states)).filter("identifier")
    einstein = EinsteinCrystal(dataset_h2[4], force_constant=1)  # different
    metrics.update(labeled, einstein)
    data_ = load_metrics(inputs=[metrics.metrics]).result()
    e_rmse = data_.e_rmse
    assert np.allclose(data.e_rmse[data.identifier != -1][:, 0], e_rmse[:, 1])
    assert np.all(~np.isnan(e_rmse[:, 0]))


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
        EMT(),
        identifier=identifier,
        error_thresholds_for_reset=(1e6, 1e6),
        error_thresholds_for_discard=(1e6, 1e6),
        metrics=Metrics(),
    )
    assert identifier.result() == 3 + len(outputs) - 2
    assert data.length().result() == len(outputs) - 2
    assert data.filter("forces").length().result() == len(outputs) - 2

    assert resets[3].result()
    assert resets[7].result()
    assert not all([r.result() for r in resets])

    identifier, data, resets = evaluate_outputs(
        outputs,
        einstein,
        EMT(),
        identifier=identifier,
        error_thresholds_for_reset=(0, 0),
        error_thresholds_for_discard=(0, 0),
        metrics=Metrics(),
    )
    assert all([r.result() for r in resets])
