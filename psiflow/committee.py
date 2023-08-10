from __future__ import annotations # necessary for type-guarding class methods
import typeguard
from typing import Optional, Union, Any
import numpy as np
import yaml

from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from parsl.app.app import python_app
from psiflow.data import Dataset, NullState
from psiflow.models import BaseModel, load_model


@typeguard.typechecked
def _compute_disagreements(
        metric: str,
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> np.ndarray:
    import numpy as np
    from psiflow.data import read_dataset, write_dataset
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
                if data[j][i] == NullState:
                    assert j == 0 # nullstates do not depend on j
                    break
                forces[j] = data[j][i].arrays['forces']
            SE = (forces - np.mean(forces, axis=0, keepdims=True)) ** 2    
            RMSE = np.sqrt(np.mean(SE))
            disagreements[i] = RMSE
    else:
        raise NotImplementedError('unknown metric ' + metric)
    return disagreements
compute_disagreements = python_app(_compute_disagreements, executors=['Default'])


# expose outside filter app to reproduce filtering behavior elsewhere
@typeguard.typechecked
def _filter_disagreements(disagreements: np.ndarray, nstates: int):
    if nstates >= len(disagreements):
        indices = np.arange(len(disagreements))
    else:
        indices = np.argsort(disagreements)[-nstates:][::-1]
    return indices
filter_disagreements = python_app(_filter_disagreements, executors=['Default'])


@typeguard.typechecked
def _extract_highest(
        disagreements: np.ndarray,
        nstates: Optional[int] = None,
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> None:
    import numpy as np
    from psiflow.data import read_dataset, write_dataset
    from psiflow.committee import _filter_disagreements
    data = read_dataset(slice(None), inputs=[inputs[0]])
    assert len(data) == len(disagreements)
    indices = _filter_disagreements(disagreements, nstates)
    write_dataset(
            [data[i] for i in indices],
            outputs=[outputs[0]],
            )
extract_highest = python_app(_extract_highest, executors=['Default'])


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

    def compute_disagreements(self, data: Dataset) -> AppFuture[np.ndarray]:
        context = psiflow.context()
        inputs = [m.evaluate(data).data_future for m in self.models]
        disagreements = compute_disagreements(
                self.metric,
                inputs=inputs,
                )
        return disagreements

    def apply(self, data: Dataset, nstates: int) -> tuple[Dataset, AppFuture]:
        disagreements = self.compute_disagreements(data)
        future = extract_highest(
                disagreements,
                nstates=nstates,
                inputs=[data.data_future],
                outputs=[psiflow.context().new_file('data_', '.xyz')],
                )
        return Dataset(None, data_future=future.outputs[0]), disagreements

    def train(self, training, validation) -> None:
        for i, model in enumerate(self.models):
            model.reset()
            model.seed += len(self.models)
        for model in self.models:
            model.initialize(training)
            model.train(training, validation)

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

