from __future__ import annotations # necessary for type-guarding class methods
import typeguard
from typing import Optional, Union, Any

from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from parsl.app.app import python_app
from psiflow.data import Dataset
from psiflow.models import BaseModel


@python_app(executors=['default'])
def apply_committee(
        discard_percentage: float,
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
    nstates = int((1 - discard_percentage) * l)
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
class CommitteeMixin:

    def __init__(
            self,
            config: Any, # can be dict or SomeModelConfig
            size: int = 4,
            discard_percentage: float = 0.5,
            metric: str = 'mean_force',
            ):
        super().__init__(config)
        self.discard_percentage = discard_percentage
        self.metric = metric

        assert size > 1
        model_cls = list(self.__class__.__bases__)[0]
        self.models = [model_cls(config) for i in range(size - 1)]
        for i, model in enumerate(self.models):
            model.seed = i + 1

    def apply(self, data: Dataset) -> tuple[Dataset, AppFuture]:
        context = psiflow.context()
        inputs = [m.evaluate(data).data_future for m in [self] + self.models]
        disagreements = apply_committee(
                self.discard_percentage,
                self.metric,
                inputs=inputs,
                outputs=[context.new_file('data_', '.xyz')],
                )
        return Dataset(None, data_future=disagreements.outputs[0]), disagreements

    def reset(self):
        super().reset()
        for i, model in enumerate(self.models):
            model.reset()
            model.seed += len(self.models) + 1

    def initialize(self, dataset: Dataset):
        super().initialize()
        for model in self.models:
            model.initialize(dataset)

    def train(self, training, validation) -> None:
        super().train(training, validation) # train first model
        for model in self.models:
            model.train(training, validation)

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ):
        super().save(path, require_done)
        for i, model in enumerate(self.models):
            model.save(path / str(i + 1), require_done)
