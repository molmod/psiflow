from typing import Optional, Union
import typeguard
from pathlib import Path
import logging

from psiflow.models import BaseModel, load_model
from psiflow.data import Dataset
from psiflow.generator import Generator, save_generators, load_generators
from psiflow.checks import Check, load_checks


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def save_state(
        path_output: Union[Path, str],
        name: str,
        model: BaseModel,
        generators: list[Generator],
        data_train: Optional[Dataset] = None,
        data_valid: Optional[Dataset] = None,
        data_failed: Optional[Dataset] = None,
        checks: Optional[list[Check]] = None,
        require_done=True,
        ) -> None:
    path = path_output / name
    path.mkdir(parents=True, exist_ok=False)

    # model
    model.save(path, require_done=require_done)

    # generators
    path_generators = path / 'generators'
    path_generators.mkdir(exist_ok=False)
    save_generators(generators, path_generators, require_done=require_done)

    # data
    if data_train is not None:
        data_train.save(path / 'train.xyz')
    if data_valid is not None:
        data_valid.save(path / 'validate.xyz')
    if data_failed is not None:
        data_failed.save(path / 'failed.xyz')

    # save checks if necessary
    if checks is not None:
        path_checks = path / 'checks'
        path_checks.mkdir(parents=False, exist_ok=False)
        for check in checks: # all checks stored in same dir
            check.save(path_checks, require_done=require_done)


@typeguard.typechecked
def load_state(path_output: Union[Path, str], name: str) -> tuple[
                BaseModel,
                list[Generator],
                Optional[Dataset],
                Optional[Dataset],
                Optional[list[Check]],
                ]:
    path = path_output / name
    assert path.is_dir() # needs to exist

    # model
    model = load_model(path)

    # generators
    path_generators = path / 'generators'
    generators = load_generators(path_generators)

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

    # checks; optional
    path_checks = path / 'checks'
    if path_checks.is_dir():
        checks = load_checks(path_checks)
    else:
        checks = None
    return model, generators, data_train, data_valid, checks
