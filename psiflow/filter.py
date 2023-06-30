from __future__ import annotations # necessary for type-guarding class methods
import typeguard
from typing import Optional, Union, Any

from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from parsl.app.app import python_app
from psiflow.data import Dataset
from psiflow.models import BaseModel


@typeguard.typechecked
class Filter:

    def apply(self, data: Dataset, nstates: int) -> Dataset:
        raise NotImplementedError


@python_app(executors=['default'])
def apply_committee(
        nstates: int,
        metric: str,
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> None:
    import numpy as np
    from psiflow.data import read_dataset, save_dataset
    data = []
    assert len(inputs) > 0
    for inp in inputs:
        _data = read_dataset(slice(None), inputs=[inp])
        data.append(_data)
    lengths = [len(d) for d in data]
    assert lengths[0] > 0
    for l in lengths:
        assert l == lengths[0]
    if nstates > lengths[0]:
        extracted = _data
    else:
        disagreements = np.zeros(lengths[0])
        if metric == 'mean_force':
            for i in range(lengths[0]):
                forces = np.zeros((len(inputs), len(data[0][0]), 3))
                for j in range(len(inputs)):
                    forces[j] = data[j][i].arrays['forces']
                SE = (forces - np.mean(forces, axis=0, keepdims=True)) ** 2    
                RMSE = np.sqrt(np.mean(SE))
                disagreements[i] = RMSE
            # extract nstates with largest disagreement
            indices = np.argsort(disagreements)[-nstates:]
            extracted = [data[0][i] for i in indices]
        else:
            raise NotImplementedError('unknown metric ' + metric)
    save_dataset(extracted, outputs=[outputs[0]])
    return disagreements


@typeguard.typechecked
class Committee(Filter):

    def __init__(
            self,
            models: list[BaseModel],
            metric: str = 'mean_force',
            ):
        for i, model in enumerate(models):
            model.seed = i
        self.models = models
        self.metric = metric

    def apply(self, data: Dataset, nstates: int) -> tuple[Dataset, AppFuture]:
        context = psiflow.context()
        inputs = [m.evaluate(data).data_future for m in self.models]
        disagreements = apply_committee(
                nstates,
                self.metric,
                inputs=inputs,
                outputs=[context.new_file('data_', '.xyz')],
                )
        return Dataset(None, data_future=disagreements.outputs[0]), disagreements

    def train(self, training, validation) -> None:
        for i, model in enumerate(self.models):
            model.reset()
            model.seed += len(self.models)
        for model in self.models:
            model.initialize(training)
            model.train(training, validation)
            model.deploy()
