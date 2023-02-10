import pytest
import numpy as np

from concurrent.futures import as_completed

from psiflow.data import FlowAtoms, Dataset
from psiflow.sampling import DynamicWalker
from psiflow.models import MACEModel
from psiflow.checks import SafetyCheck
from psiflow.reference import EMTReference
from psiflow.generator import Generator


def test_generator_mace(context, dataset, mace_config):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    reference = EMTReference(context)

    model = MACEModel(context, mace_config)
    model.initialize(dataset[:3])
    model.deploy()
    checks = [SafetyCheck()]

    generator = Generator('bla', walker, reference, None)
    state = generator(model, checks, wait_for_it=[])
    assert checks[0].nchecks == 0
    state.result()
    assert checks[0].nchecks == 1
    assert checks[0].npasses.result() == 1

    generator.walker.tag_unsafe() # will retry internally
    state = generator(model, checks, wait_for_it=[])
    state.result() # finish everything
    assert checks[0].nchecks == 3
    assert checks[0].npasses.result() == 2

    # train model and generate afterwards
    old = model.deploy_future['float32'].filepath
    model.train(dataset[:5], dataset[5:7])
    assert len(model.deploy_future) == 0
    with pytest.raises(KeyError):
        state = generator(model, checks, wait_for_it=[])
        state.result() # force KeyError
    model.deploy()
    new = model.deploy_future['float32'].filepath
    assert old != new
    state = generator(model, checks, wait_for_it=[])
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
    state = generator(model, checks, wait_for_it=[])
    for i, future in enumerate(as_completed([state, model.model_future])):
        if i == 0: # first, the propagation finishes
            assert isinstance(future.result(), FlowAtoms)
        else: # then, the model finishes training
            assert not isinstance(future.result(), FlowAtoms)


def test_generator_multiply(context, dataset, mace_config):
    walker = DynamicWalker(context, dataset[0], steps=10, step=1)
    reference = EMTReference(context)

    model = MACEModel(context, mace_config)
    model.initialize(dataset[:3])
    model.deploy()

    generator = Generator('bla', walker, reference, None)
    generators = generator.multiply(10)
    data = Dataset(context, [g(model) for g in generators])

    # no two states should be the same
    for i in range(data.length().result() - 1):
        for j in range(i + 1, data.length().result()):
            assert not np.allclose(
                    data[i].result().get_positions(),
                    data[j].result().get_positions(),
                    )

