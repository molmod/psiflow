from pathlib import Path

from psiflow.models import NequIPModel
from psiflow.sampling import RandomWalker, DynamicWalker
from psiflow.state import save_state, load_state


def test_save_load(context, dataset, nequip_config, tmp_path):
    model = NequIPModel(nequip_config)
    path_output = Path(tmp_path) / 'output'
    path_output.mkdir()
    walkers = RandomWalker.multiply(2, dataset) + DynamicWalker.multiply(2, dataset)
    name = 'test'
    save_state(
            path_output=path_output,
            name=name,
            model=model,
            walkers=walkers,
            data_train=dataset,
            )
    assert (path_output / name / 'walkers').is_dir()
    assert (path_output / name / 'walkers' / '0').is_dir()
    assert (path_output / name / 'walkers' / '1').is_dir()
    assert (path_output / name / 'walkers' / '2').is_dir()
    assert (path_output / name / 'walkers' / '3').is_dir()
    assert (path_output / name / 'train.xyz').is_file()

    model_, walkers_, data_train, data_valid = load_state(path_output, 'test')
    assert model_.config_future is None # model was not initialized
    assert len(walkers_) == 4
    assert data_train.length().result() == 0
    assert data_valid.length().result() == 0
    model.initialize(dataset[:2])
    model.deploy()
    name = 'test_'
    save_state(path_output, name, model=model, walkers=walkers)
    model_, walkers_, data_train, data_valid = load_state(path_output, 'test_')
    assert model_.config_future is not None # model was initialized
