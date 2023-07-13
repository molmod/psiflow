from __future__ import annotations # necessary for type-guarding class methods
import typeguard
from typing import Optional, Union, Any
import numpy as np
import yaml

from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from parsl.app.app import python_app
from psiflow.data import Dataset
from psiflow.models import BaseModel, load_model


@python_app(executors=['default'])
def compute_disagreements(
        metric: str,
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> np.ndarray:
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
    disagreements = np.zeros(lengths[0])
    if metric == 'mean_force':
        for i in range(lengths[0]):
            forces = np.zeros((len(inputs), len(data[0][0]), 3))
            for j in range(len(inputs)):
                forces[j] = data[j][i].arrays['forces']
            SE = (forces - np.mean(forces, axis=0, keepdims=True)) ** 2    
            RMSE = np.sqrt(np.mean(SE))
            disagreements[i] = RMSE
    else:
        raise NotImplementedError('unknown metric ' + metric)
    return disagreements


@python_app(executors=['default'])
def filter_disagreements(
        disagreements: np.ndarray,
        retain_percentage: Optional[float] = None,
        nstates: Optional[int] = None,
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> None:
    import numpy as np
    from psiflow.data import read_dataset, save_dataset
    if nstates is None:
        assert retain_percentage is not None
        nstates = int(retain_percentage) * len(disagreements)
    else:
        assert retain_percentage is None
    indices = np.argsort(disagreements)[-nstates:][::-1]
    data = read_dataset(slice(None), inputs=[inputs[0]])
    assert len(data) == len(disagreements)
    save_dataset(
            read_dataset(
                [int(i) for i in indices],
                inputs=[inputs[0]],
                ),
            outputs=[outputs[0]],
            )


@typeguard.typechecked
class Committee:

    def __init__(
            self,
            models: list[BaseModel],
            metric: str = 'mean_force',
            ):
        self.models = models
        self.metric = metric
        for i, model in enumerate(self.models):
            model.seed = i

    def compute_disagreements(self, data: Dataset) -> AppFuture:
        context = psiflow.context()
        inputs = [m.evaluate(data).data_future for m in self.models]
        disagreements = compute_disagreements(
                self.metric,
                inputs=inputs,
                )
        return disagreements

    def apply(self, data: Dataset, nstates_or_retain: Union[int, float]) -> Dataset:
        disagreements = self.compute_disagreements(data)
        if type(nstates_or_retain) == float:
            assert nstates_or_retain <= 1.0
            assert nstates_or_retain >  0.0
            retain_percentage = nstates_or_retain
            nstates = None
        else:
            nstates = nstates_or_retain
            retain_percentage = None
        future = filter_disagreements(
                disagreements,
                retain_percentage=retain_percentage,
                nstates=nstates,
                inputs=[data.data_future],
                outputs=[psiflow.context().new_file('data_', '.xyz')],
                )
        return Dataset(None, data_future=future.outputs[0])

    def train(self, training, validation) -> None:
        for i, model in enumerate(self.models):
            model.reset()
            model.seed += len(self.models)
        for model in self.models:
            model.initialize(training)
            model.train(training, validation)
            model.deploy()

    def save(
            self,
            path: Union[Path, str],
            require_done: bool = True,
            ):
        for i, model in enumerate(self.models):
            model.save(path / str(i), require_done)

    @classmethod
    def load(cls, path: Union[Path, str]) -> Committee:
        path = Path(path)
        assert path.exists()
        assert (path / 'Committee.yaml').exists()
        with open(path / 'Committee.yaml', 'r') as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        models = []
        i = 0
        while (path / str(i)).exists():
            models.append(load_model(path / str(i)))
            i += 1
        return cls(models, **kwargs)

