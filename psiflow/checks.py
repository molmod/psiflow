from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Dict
import typeguard
from pathlib import Path
import glob
import yaml
import numpy as np

from ase import Atoms

from parsl.app.app import python_app, join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

from psiflow.execution import ExecutionContext
from psiflow.models import BaseModel
from psiflow.data import Dataset, FlowAtoms
from psiflow.utils import save_yaml, copy_app_future
from psiflow.models import load_model


@join_app
@typeguard.typechecked
def apply_check(
        check: Check,
        state: Union[FlowAtoms, Atoms],
        checked_state: Optional[Union[FlowAtoms, Atoms]],
        ) -> AppFuture:
    check.nchecks += 1
    if checked_state is not None:
        check.npasses += 1
    else:
        check.states.append(state)
    return copy_app_future(checked_state)


@typeguard.typechecked
class Check:

    def __init__(self) -> None:
        self.nchecks = 0
        self.npasses = 0
        self.states  = []

    def __call__(
            self,
            state: AppFuture,
            tag: Optional[AppFuture] = None,
            ) -> AppFuture:
        return apply_check(
                self,
                state,
                self._apply_check(state, tag),
                )

    def reset(self) -> None:
        self.nchecks = 0
        self.npasses = 0

    def save(self, path: Union[Path, str], require_done: bool = True) -> DataFuture:
        path = Path(path)
        assert path.is_dir()
        future = save_yaml(
                self.parameters, # property which returns dict of parameters
                outputs=[File(str(path / (self.__class__.__name__ + '.yaml')))],
                ).outputs[0]
        if require_done:
            future.result()
        return future

    def _apply_check(self):
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[Path, str], *args: Any):
        path = Path(path)
        assert path.is_dir()
        path_pars = path / (cls.__name__ + '.yaml')
        assert path_pars.is_file()
        with open(path_pars, 'r') as f:
            pars_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**pars_dict)

    @property
    def parameters(self) -> Dict:
        return {}


@typeguard.typechecked
def _check_distances(
        state: Optional[FlowAtoms],
        threshold: float,
        ) -> Optional[FlowAtoms]:
    import numpy as np
    from ase.geometry.geometry import find_mic
    if state is None:
        return None
    nrows = int(len(state) * (len(state) - 1) / 2)
    deltas = np.zeros((nrows, 3))
    count = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            deltas[count] = state.positions[i] - state.positions[j]
            count += 1
    assert count == nrows
    deltas, _ = find_mic(deltas, state.cell)
    check = np.all(np.linalg.norm(deltas, axis=1) > threshold)
    if check:
        return state
    else:
        return None
check_distances = python_app(_check_distances, executors=['default'])


@typeguard.typechecked
class InteratomicDistanceCheck(Check):

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def _apply_check(
            self,
            state: AppFuture,
            tag: Optional[AppFuture] = None,
            ) -> AppFuture:
        return check_distances(state, self.threshold)

    @property
    def parameters(self) -> Dict:
        return {'threshold': self.threshold}


@typeguard.typechecked
def _check_discrepancy(
        state: Optional[FlowAtoms],
        errors: np.ndarray,
        thresholds: List[float],
        ) -> Optional[FlowAtoms]:
    if state is None:
        return None
    assert len(thresholds) == errors.shape[1]
    assert errors.shape[0] == 1
    check = False
    for j in range(len(thresholds)):
        if errors[0, j] > thresholds[j]:
            check = True # check passes when threshold is exceeded
    if check:
        return state
    else:
        return None
check_discrepancy = python_app(_check_discrepancy, executors=['default'])


@typeguard.typechecked
class DiscrepancyCheck(Check):

    def __init__(
            self,
            metric: str,
            properties: List[str],
            thresholds: List[float],
            model_old: Optional[BaseModel] = None,
            model_new: Optional[BaseModel] = None,
            ) -> None:
        super().__init__()
        self.metric = metric
        assert len(properties) == len(thresholds)
        self.properties = properties
        self.thresholds = thresholds
        if model_old is not None:
            assert len(model_old.deploy_future) > 0
        if model_new is not None:
            assert len(model_new.deploy_future) > 0
        self.model_old = model_old # can be initialized with None models
        self.model_new = model_new

    def _apply_check(
            self,
            state: AppFuture,
            tag: Optional[AppFuture] = None,
            ) -> AppFuture:
        assert self.model_old is not None
        assert self.model_old.config_future is not None
        assert self.model_new is not None
        assert self.model_new.config_future is not None
        dataset = Dataset(self.model_old.context, [state]) # dummy
        errors = Dataset.get_errors(
                self.model_old.evaluate(dataset),
                self.model_new.evaluate(dataset),
                metric=self.metric,
                properties=self.properties,
                )
        return check_discrepancy(state, errors, self.thresholds)

    def update_model(self, model: BaseModel) -> None:
        assert model.config_future is not None # initialized
        assert len(model.deploy_future) > 0 # and deployed
        self.model_old = self.model_new
        self.model_new = model

    def save(self, path: Union[Path, str], require_done: bool = True) -> None:
        path = Path(path)
        super().save(path, require_done)
        if self.model_old is not None:
            path_old = path / 'model_old'
            path_old.mkdir(parents=False, exist_ok=False)
            self.model_old.save(path_old, require_done=require_done)
        if self.model_new is not None:
            path_new = path / 'model_new'
            path_new.mkdir(parents=False, exist_ok=False)
            self.model_new.save(path_new, require_done=require_done)

    @classmethod
    def load(cls, path: Union[Path, str], context: ExecutionContext) -> DiscrepancyCheck:
        check = super(DiscrepancyCheck, cls).load(path)
        path_old = path / 'model_old'
        if path_old.is_dir():
            model = load_model(context, path_old)
            model.deploy()
            check.update_model(model)
        path_new = path / 'model_new'
        if path_new.is_dir():
            model = load_model(context, path_new)
            model.deploy()
            check.update_model(model)
        return check

    @property
    def parameters(self) -> Dict:
        return {
                'metric': self.metric,
                'properties': list(self.properties),
                'thresholds': list(self.thresholds),
                }


@typeguard.typechecked
def _check_safety(state: Optional[FlowAtoms], tag: str):
    if state is None:
        return None
    if tag == 'unsafe':
        return None
    else:
        return state
check_safety = python_app(_check_safety, executors=['default'])


@typeguard.typechecked
class SafetyCheck(Check):

    def _apply_check(
            self,
            state: AppFuture,
            tag: AppFuture,
            ) -> AppFuture:
        return check_safety(state, tag)


@typeguard.typechecked
def load_checks(path: Union[Path, str], context: ExecutionContext) -> Optional[List[Check]]:
    path = Path(path)
    assert path.is_dir()
    checks = []
    classes = [
            InteratomicDistanceCheck,
            DiscrepancyCheck,
            SafetyCheck,
            None,
            ]
    for filename in glob.glob(str(path) + '/*.yaml'):
        for check_cls in classes:
            assert check_cls is not None
            if Path(filename).stem == check_cls.__name__:
                break
        checks.append(check_cls.load(path, context))
    if len(checks) == 0:
        return None
    return checks
