from pathlib import Path

from psiflow.models import NequIPModel
from psiflow.generator import Generator
from psiflow.sampling import RandomWalker, DynamicWalker
from psiflow.checks import SafetyCheck, DiscrepancyCheck
from psiflow.state import save_state, load_state


def test_save_load(context, dataset, nequip_config, tmp_path):
    model = NequIPModel(nequip_config)
    path_output = Path(tmp_path) / 'output'
    path_output.mkdir()
    generators = (
            Generator('random', RandomWalker(dataset[0])).multiply(2) +
            Generator('dynamic', DynamicWalker(dataset[1])).multiply(2)
            )
    checks = [
            SafetyCheck(),
            DiscrepancyCheck(
                metric='mae',
                properties=['energy'],
                thresholds=[1],
                model_old=None,
                model_new=None,
                ),
            ]
    name = 'test'
    save_state(
            path_output=path_output,
            name=name,
            model=model,
            generators=generators,
            checks=checks,
            data_failed=dataset,
            )
    assert (path_output / name / 'generators').is_dir()
    assert (path_output / name / 'generators' / 'random0').is_dir()
    assert (path_output / name / 'generators' / 'random1').is_dir()
    assert (path_output / name / 'generators' / 'dynamic0').is_dir()
    assert (path_output / name / 'generators' / 'dynamic1').is_dir()
    assert not (path_output / name / 'generators' / '0').is_dir()
    assert (path_output / name / 'checks').is_dir() # directory for saved checks
    assert (path_output / name / 'failed.xyz').is_file()

    model_, generators_, data_train, data_valid, checks = load_state(path_output, 'test')
    assert model_.config_future is None # model was not initialized
    assert len(generators_) == 4
    assert data_train.length().result() == 0
    assert data_valid.length().result() == 0
    assert len(checks) == 2
    model.initialize(dataset[:2])
    model.deploy()
    checks = [
            SafetyCheck(),
            DiscrepancyCheck(
                metric='mae',
                properties=['energy'],
                thresholds=[1],
                model_old=model,
                model_new=model.copy(),
                ),
            ]
    name = 'test_'
    save_state(path_output, name, model=model, generators=generators, checks=checks)
    model_, generators_, data_train, data_valid, checks = load_state(path_output, 'test_')
    assert model_.config_future is not None # model was initialized
    for check in checks: # order is arbitrary
        if type(check) == DiscrepancyCheck:
            assert check.model_old is not None
            assert check.model_new is not None
