from pathlib import Path
import glob
import yaml

from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.data import Dataset
from flower.utils import save_yaml
from flower.models import load_model


class Check:

    def __call__(self, state, tag=None):
        raise NotImplementedError

    def save(self, path, require_done=True):
        path = Path(path)
        assert path.is_dir()
        future = save_yaml(
                self.parameters, # property which returns dict of parameters
                outputs=[File(str(path / (self.__class__.__name__ + '.yaml')))],
                ).outputs[0]
        future.result()
        return future

    @classmethod
    def load(cls, path, *args): # requires a context in some cases
        path = Path(path)
        assert path.is_dir()
        path_pars = path / (cls.__name__ + '.yaml')
        assert path_pars.is_file()
        with open(path_pars, 'r') as f:
            pars_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**pars_dict)

    @property
    def parameters(self):
        return {}


@python_app
def check_distances(state, threshold):
    import numpy as np
    from ase.geometry.geometry import find_mic
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


class InteratomicDistanceCheck(Check):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, state, tag=None):
        return check_distances(state, self.threshold)

    @property
    def parameters(self):
        return {'threshold': self.threshold}


@python_app
def check_discrepancy(state, errors, thresholds):
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


class DiscrepancyCheck(Check):

    def __init__(self, metric, properties, thresholds, model_old=None, model_new=None):
        self.metric = metric
        assert type(properties) == list
        assert type(thresholds) == list
        assert len(properties) == len(thresholds)
        self.properties = properties
        self.thresholds = thresholds
        if model_old is not None:
            assert len(model_old.deploy_future) > 0
        if model_new is not None:
            assert len(model_new.deploy_future) > 0
        self.model_old = model_old # can be initialized with None models
        self.model_new = model_new

    def __call__(self, state, tag=None):
        assert self.model_old is not None
        assert self.model_old.config_future is not None
        assert self.model_new is not None
        assert self.model_new.config_future is not None
        dataset = Dataset(self.model_old.context, atoms_list=[state])
        dataset = self.model_old.evaluate(dataset, suffix='_old')
        dataset = self.model_new.evaluate(dataset, suffix='_new')
        errors = dataset.get_errors(
                intrinsic=False,
                metric=self.metric,
                suffix_0='_old',
                suffix_1='_new',
                properties=self.properties,
                )
        return check_discrepancy(state, errors, self.thresholds)

    def update_model(self, model):
        assert model is not None
        assert model.config_future is not None
        assert len(model.deploy_future) > 0
        self.model_old = self.model_new
        self.model_new = model

    def save(self, path, require_done=True):
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
    def load(cls, path, context):
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
    def parameters(self):
        return {
                'metric': self.metric,
                'properties': list(self.properties),
                'thresholds': list(self.thresholds),
                }


@python_app
def check_safety(state, tag):
    if tag == 'unsafe':
        return None
    else:
        return state


class SafetyCheck(Check):

    def __call__(self, state, tag):
        return check_safety(state, tag)


def load_checks(path, context):
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
