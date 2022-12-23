from parsl.app.app import python_app, join_app

from flower.data import Dataset


class Check:

    def __call__(self, state, tag=None):
        raise NotImplementedError


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

    def __init__(self, model_old, model_new, metric, properties, thresholds):
        self.model_old = model_old
        self.model_new = model_new
        self.metric = metric
        assert len(properties) == len(thresholds)
        self.properties = properties
        self.thresholds = thresholds


    def __call__(self, state, tag=None):
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


@python_app
def check_safety(state, tag):
    if tag == 'unsafe':
        return None
    else:
        return state


class SafetyCheck(Check):

    def __call__(self, state, tag):
        return check_safety(state, tag)
