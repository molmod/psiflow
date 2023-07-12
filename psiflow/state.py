from typing import Optional, Union
import typeguard
from pathlib import Path
import logging

from psiflow.models import BaseModel, load_model
from psiflow.data import Dataset
from psiflow.walkers import BaseWalker, save_walkers, load_walkers
from psiflow.utils import resolve_and_check


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def save_state(
        path_output: Union[Path, str],
        name: str,
        model: BaseModel,
        walkers: list[BaseWalker],
        data_train: Optional[Dataset] = None,
        data_valid: Optional[Dataset] = None,
        require_done=True,
        ) -> None:
    path_output = resolve_and_check(path_output)
    path = path_output / name
    path.mkdir(parents=True, exist_ok=False)

    # model
    model.save(path, require_done=require_done)

    # walkers
    path_walkers = path / 'walkers'
    path_walkers.mkdir(exist_ok=False)
    save_walkers(walkers, path_walkers, require_done=require_done)

    # data
    if data_train is not None:
        data_train.save(path / 'train.xyz')
    if data_valid is not None:
        data_valid.save(path / 'validate.xyz')


@typeguard.typechecked
def load_state(path_output: Union[Path, str], name: str) -> tuple[
                BaseModel,
                list[BaseWalker],
                Optional[Dataset],
                Optional[Dataset],
                ]:
    path_output = resolve_and_check(path_output)
    path = path_output / name
    assert path.is_dir() # needs to exist

    # model
    model = load_model(path)

    # walkers
    path_walkers = path / 'walkers'
    walkers = load_walkers(path_walkers)

    # data; optional
    path_train = path / 'train.xyz'
    if path_train.is_file():
        data_train = Dataset.load(path_train)
    else:
        data_train = Dataset([]) # empty dataset
    path_valid = path / 'validate.xyz'
    if path_valid.is_file():
        data_valid = Dataset.load(path_valid)
    else:
        data_valid = Dataset([])
    return model, walkers, data_train, data_valid
