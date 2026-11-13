import numpy as np

import psiflow
from psiflow.data import read_frames
from psiflow.hamiltonians import EinsteinCrystal
from psiflow.reference import ReferenceDummy
from psiflow.sampling import Walker
from psiflow.sampling.output import Status
from psiflow.models import MACE
from psiflow.learning import (
    Learning,
    FILE_MODEL,
    FILE_SAMPLE,
    FILE_REFERENCE,
    FILE_DATA,
    FILE_TRAIN,
    FILE_VAL,
    FILE_RECORDS,
    check_learning_iteration,
    collect_geometries,
    compare_geometries,
    analyze_outputs,
    ErrorThresholds,
)
from psiflow.utils.apps import pack


def test_learning_thresholds(tmp_path, dataset):
    """"""
    geom = dataset[0]
    mace = MACE(tmp_path, {})
    reference = ReferenceDummy()
    learning = Learning(mace, reference)
    walkers = Walker(geom).multiply(10)
    hamiltonian = EinsteinCrystal.from_geometry(geom, 1)

    # manually perform part of the learning loop
    execute = learning._next_iteration("passive_learning")
    data, outputs = learning._sample(hamiltonian, walkers, steps=10)
    geoms_ref = reference.evaluate(data).geometries()
    geoms_mod = hamiltonian.evaluate(data).geometries()

    # no resets or discards
    future = compare_geometries(geoms_ref, geoms_mod, None)
    data, records = future.result()
    assert len(data) == len(records) == 10
    assert not any(r.reset or r.discard for r in records)

    # force failed single-points
    geoms = geoms_ref.result()
    geoms[0].reset()
    geoms[5].reset()
    future = compare_geometries(geoms, geoms_mod, None)
    data, records = future.result()
    assert len(data) == 8
    assert len(records) == 10
    assert records[0].reset == records[0].discard == True
    assert records[5].reset == records[5].discard == True
    assert sum(r.reset or r.discard for r in records) == 2

    # check error thresholds -- only resets
    thresholds = ErrorThresholds(0, np.inf, 0, np.inf)
    future = compare_geometries(geoms, geoms_mod, thresholds)
    data, records = future.result()
    assert len(data) == 8
    assert len(records) == 10
    assert all(r.reset for r in records)
    assert sum(r.discard for r in records) == 2

    # check error thresholds -- only discards
    thresholds = ErrorThresholds(np.inf, 0, np.inf, 0)
    future = compare_geometries(geoms, geoms_mod, thresholds)
    data, records = future.result()
    assert len(data) == 0
    assert len(records) == 10
    assert sum(r.reset for r in records) == 2
    assert all(r.discard for r in records)


def test_learning_reset(tmp_path, dataset):
    """"""
    mace = MACE(tmp_path, {})
    reference = ReferenceDummy()
    learning = Learning(mace, reference, initial_data=dataset + dataset)

    learning.wait_for.result()
    train = collect_geometries(learning.root, FILE_TRAIN).result()
    val = collect_geometries(learning.root, FILE_VAL).result()

    learning._reset_training()
    learning.wait_for.result()
    train_ = collect_geometries(learning.root, FILE_TRAIN).result()
    val_ = collect_geometries(learning.root, FILE_VAL).result()

    assert len(train) == len(train_)
    assert len(val) == len(val_)
    assert train != train_
    assert val != val_


def test_learning_analyze(tmp_path, dataset):
    """"""
    geom = dataset[0]
    mace = MACE(tmp_path, {})
    reference = ReferenceDummy()
    learning = Learning(mace, reference)
    walkers = Walker(geom).multiply(10)
    hamiltonian = EinsteinCrystal.from_geometry(geom, 1)

    # manually perform part of the learning loop
    execute = learning._next_iteration("passive_learning")
    data, outputs = learning._sample(hamiltonian, walkers, steps=10)
    records = learning._evaluate(data, hamiltonian)
    records = records.result()
    statuses = pack(*[o.status for o in outputs]).result()
    temperature = pack(*[o.temperature for o in outputs]).result()
    time = pack(*[o.time for o in outputs]).result()

    # no resets
    future = analyze_outputs(records, statuses, temperature, time)
    walker_records, reset_mask = future.result()
    assert len(walker_records) == len(reset_mask) == 10
    assert not any(r.reset for r in walker_records)
    assert not any(reset_mask)

    # force some resets
    records[0].reset = records[-1].reset = True
    future = analyze_outputs(records, statuses, temperature, time)
    walker_records, reset_mask = future.result()
    assert len(walker_records) == len(reset_mask) == 10
    assert reset_mask[0] == reset_mask[-1] == True
    assert sum(r.reset for r in walker_records) == 2

    # force more resets -- timeout is fine
    statuses = list(statuses)
    statuses[1:4] = [Status.TIMEOUT, Status.EXPLODED, Status.FORCE_EXCEEDED]
    future = analyze_outputs(records, statuses, temperature, time)
    walker_records, reset_mask = future.result()
    assert len(walker_records) == len(reset_mask) == 10
    assert all(reset_mask[i] for i in (0, 2, 3, -1))
    assert sum(r.reset for r in walker_records) == 4


def test_learning_workflow(tmp_path, gpu, mace_config, dataset):
    """"""
    geom = dataset[0]
    n = 5
    hamiltonian = EinsteinCrystal.from_geometry(geom, 1)
    mace = MACE(tmp_path, mace_config)
    walkers = Walker(geom).multiply(n)
    walkers[1].temperature = 200
    walkers[2].timestep = 1

    # check basic functionality
    learning = Learning(mace, ReferenceDummy(), initial_data=dataset)
    identifier = learning.identifier.result()
    assert learning.iteration == -1
    assert learning.workdir is None
    assert identifier == dataset.length().result()

    walkers = learning.passive_learning(hamiltonian, walkers, 20)
    assert learning.iteration == 0
    assert learning.workdir == learning.root / "0_passive_learning"
    assert learning.identifier.result() == identifier + n
    assert learning.model.iteration == 1  # init + train

    walkers = learning.active_learning(walkers, 20)
    assert learning.iteration == 1
    assert learning.workdir == learning.root / "1_active_learning"
    assert learning.identifier.result() == identifier + 2 * n
    assert learning.model.iteration == 2

    psiflow.wait()
    assert not any([w.is_reset().result() for w in walkers])

    # check restart
    learning = Learning(mace, ReferenceDummy(), initial_data=dataset)
    assert learning.identifier == identifier + 2 * n
    assert mace.model_future.filepath == str(
        learning.root / "1_active_learning" / FILE_MODEL
    )

    # existing iterations should be skipped
    walkers_ = learning.passive_learning(hamiltonian, walkers, 20)
    walkers_ = learning.active_learning(walkers, 20)
    assert learning.wait_for is None
    assert learning.iteration == 1
    assert learning.workdir == learning.root / "1_active_learning"
    assert learning.model.iteration == 2
    for w, w_ in zip(walkers, walkers_):
        # skipped iterations should consistently return walkers
        variables = vars(w).copy()
        variables_ = vars(w_).copy()
        start, start_ = variables.pop("start"), variables_.pop("start")
        assert start.result() == start_
        state, state_ = variables.pop("state"), variables_.pop("state")
        assert state.result() == state_
        assert variables == variables_

    walkers_ = learning.active_learning(walkers_, 20, reset_training=True)
    assert learning.iteration == 2
    assert learning.workdir == learning.root / "2_active_learning"
    assert learning.identifier.result() == identifier + 3 * n
    assert learning.model.iteration == 4  # init + train

    psiflow.wait()

    # check contents of output directories
    history = learning._get_learning_history()
    for i, name in history.items():
        assert check_learning_iteration(learning.root / name)  # all files exist?

    # inspect final iteration -- no discarded geometries
    workdir = learning.workdir
    geoms_sample = read_frames(workdir / FILE_SAMPLE).result()
    geoms_ref = read_frames(workdir / FILE_REFERENCE).result()
    geoms_data = read_frames(workdir / FILE_DATA).result()
    assert len(geoms_sample) == len(geoms_ref) == len(geoms_data) == n
    text = (workdir / FILE_RECORDS).read_text()
    records_dict = psiflow.deserialize(text).result()
    assert len(records_dict["records"]) == n
    assert len(records_dict["walker_records"]) == n

    # check all training data
    geoms = collect_geometries(learning.root, "*data.xyz").result()
    train = collect_geometries(learning.root, FILE_TRAIN).result()
    val = collect_geometries(learning.root, FILE_VAL).result()
    assert len(geoms) == dataset.length().result() + 3 * n
    assert len(geoms) == len(train) + len(val)
