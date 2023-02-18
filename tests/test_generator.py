import pytest
import numpy as np
import os

from concurrent.futures import as_completed

from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import DynamicWalker
from psiflow.models import MACEModel
from psiflow.checks import SafetyCheck
from psiflow.reference import EMTReference
from psiflow.generator import Generator, save_generators, load_generators


def test_generator_mace(context, dataset, mace_config):
    walker = DynamicWalker(dataset[0], steps=10, step=1)
    reference = EMTReference()

    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    checks = [SafetyCheck()]

    generator = Generator('bla', walker, None)
    state = generator(model, reference, checks=checks, wait_for_it=[])
    assert checks[0].nchecks == 0
    state.result()
    assert checks[0].nchecks == 1
    assert checks[0].npasses == 1

    generator.walker.tag_unsafe() # will retry internally
    state = generator(model, reference, checks=checks, wait_for_it=[])
    state.result() # finish everything
    assert checks[0].nchecks == 3
    assert checks[0].npasses == 2

    # test retry mechanism
    generator.nretries_sampling = 0
    generator.walker.tag_unsafe()
    state = generator(model, None, checks)
    assert state.result().reference_status == False
    assert np.allclose(
            state.result().get_positions(),
            generator.walker.start_future.result().get_positions(),
            )
    assert generator.walker.counter_future.result() == 0 # walker was reset

    generator.nretries_sampling = 1
    generator.walker.tag_unsafe()
    state = generator(model, reference, checks)
    assert state.result().reference_status == True
    assert not np.allclose(
            state.result().get_positions(),
            generator.walker.start_future.result().get_positions(),
            )
    assert generator.walker.counter_future.result() != 0 # walker propagated

    # check whether reference energy/forces/stress are saved
    generated = Dataset([state])
    errors = Dataset.get_errors(
            generated,
            reference.evaluate(generated),
            ).result()
    assert np.allclose(errors, 0, atol=1e-3)

    # use generator without reference
    state = generator(model, None, checks)
    generated = Dataset([state])
    errors = Dataset.get_errors(
            generated,
            reference.evaluate(generated),
            ).result()
    assert not np.allclose(errors, 0, atol=1e-3)

    # train model and generate afterwards
    old = model.deploy_future['float32'].filepath
    model.train(dataset[:5], dataset[5:7])
    assert len(model.deploy_future) == 0
    with pytest.raises(KeyError):
        state = generator(model, reference, checks, wait_for_it=[])
        state.result() # force KeyError
    model.deploy()
    new = model.deploy_future['float32'].filepath
    assert old != new
    state = generator(model, reference, checks, wait_for_it=[])
    for i, future in enumerate(as_completed([state, model.model_future])):
        if i == 0: # first, the model finishes training
            assert not isinstance(future.result(), FlowAtoms)
        else: # then, the propagation completes
            assert isinstance(future.result(), FlowAtoms)

    # train model and generate simultaneously
    old = model.deploy_future['float32'].filepath
    model.train(dataset[:5], dataset[5:7], keep_deployed=True)
    assert len(model.deploy_future) != 0
    new = model.deploy_future['float32'].filepath
    assert old == new
    state = generator(model, reference, checks, wait_for_it=[])
    for i, future in enumerate(as_completed([state, model.model_future])):
        if i == 0: # first, the propagation finishes
            assert isinstance(future.result(), FlowAtoms)
        else: # then, the model finishes training
            assert not isinstance(future.result(), FlowAtoms)


def test_generator_multiply(context, dataset, mace_config, tmp_path):
    walker = DynamicWalker(dataset[0], steps=10, step=1)
    reference = EMTReference()

    model = MACEModel(mace_config)
    model.initialize(dataset[:3])
    model.deploy()

    generator = Generator('bla', walker, None)
    generators = generator.multiply(10, initialize_using=dataset)
    for i, generator in enumerate(generators):
        assert np.allclose(
                generator.walker.state_future.result().get_positions(),
                dataset[i].result().get_positions(),
                )
    data = Dataset([g(model, None) for g in generators])

    # no two states should be the same
    for i in range(data.length().result() - 1):
        for j in range(i + 1, data.length().result()):
            assert not np.allclose(
                    data[i].result().get_positions(),
                    data[j].result().get_positions(),
                    )

    # test save and load
    save_generators(generators, tmp_path)
    _generators = load_generators(tmp_path)
    ndirs = len([f for f in os.listdir(tmp_path) if os.path.isdir(tmp_path / f)])
    assert ndirs == len(generators)

    states = Dataset([_g.walker.state_future for _g in _generators])
    for g in generators:
        found = False
        for i in range(states.length().result()):
            if np.allclose(
                    states[i].result().get_positions(),
                    g.walker.state_future.result().get_positions(),
                    ):
                assert not found
                found = True
        assert found

    # test with multiple checks
    generators[0].walker.tag_unsafe()
    generators[1].walker.tag_unsafe()
    checks = [SafetyCheck()]
    states = [g(model, reference, checks) for g in generators]
    dataset = Dataset(states)
    dataset.length().result()
    assert len(checks[0].states) == 2
