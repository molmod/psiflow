import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from flower.models import BaseModel, load_model
from flower.reference.base import BaseReference
from flower.sampling import RandomWalker
from flower.ensemble import Ensemble
from flower.data import Dataset
from flower.checks import Check, load_checks


class Manager:

    def __init__(self, path_output):
        self.path_output = Path(path_output)
        self.path_output.mkdir(parents=True, exist_ok=True)

    def dry_run(
            self,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble = None,
            random_walker: RandomWalker = None,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ):
        context = model.context
        if random_walker is None:
            assert ensemble is not None
            random_walker = RandomWalker( # with default parameters
                    context,
                    ensemble.walkers[0].start_future,
                    )

        # single point evaluation
        evaluated = reference.evaluate(random_walker.state_future)
        evaluated.result()

        # generation of small dataset
        _ensemble = Ensemble.from_walker(random_walker, nwalkers=5)
        data = _ensemble.propagate(7, model=None, checks=[])
        data = reference.evaluate(data)

        # short training and deploy
        model.reset()
        max_epochs = model.config_raw['max_epochs']
        model.config_raw['max_epochs'] = 2
        if (data_train is not None) and (data_valid is not None):
            assert data_train.length.result() >= 5 # only pass nonempty data
            assert data_valid.length.result() >= 2
            model.initialize(data_train)
            model.train(data_train, data_valid)
        new_train = data[:5]
        new_valid = data[5:]
        model.reset()
        model.initialize(new_train)
        model.train(new_train, new_valid)
        model.config_raw['max_epochs'] = max_epochs # revert to old max_epochs

        # deploy and propagate ensemble
        model.deploy()
        if ensemble is not None:
            data = ensemble.propagate(
                    ensemble.nwalkers,
                    checks=checks,
                    model=model,
                    )
            data = reference.evaluate(data)
            assert data.length.result() == ensemble.nwalkers

        # save objects
        if ensemble is None:
            ensemble = _ensemble
        self.save( # test save
                prefix='dry_run',
                model=model,
                ensemble=ensemble,
                data_train=new_train,
                data_valid=new_valid,
                checks=checks,
                )

    def save(
            self,
            prefix,
            model: BaseModel,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ):
        path = self.path_output / prefix
        path.mkdir(parents=False, exist_ok=False) # parent should exist

        # model
        model.save(path)

        # ensemble
        path_ensemble = path / 'ensemble'
        path_ensemble.mkdir(parents=False)
        ensemble.save(path_ensemble)

        # data
        if data_train is not None:
            data_train.save(path / 'train.xyz')
        if data_valid is not None:
            data_valid.save(path / 'validate.xyz')

        # save checks if necessary
        if checks is not None:
            path_checks = path / 'checks'
            path_checks.mkdir(parents=False)
            for check in checks:
                check.save(path_checks) # all checks may be stored in same dir

    def load(self, prefix, context):
        path = self.path_output / prefix
        assert path.is_dir() # needs to exist

        # model
        model = load_model(context, path)

        # ensemble
        path_ensemble = path / 'ensemble'
        ensemble = Ensemble.load(context, path_ensemble)

        # data; optional
        path_train = path / 'train.xyz'
        if path_train.is_file():
            data_train = Dataset.load(context, path_train)
        else:
            data_train = Dataset(context)
        path_valid = path / 'validate.xyz'
        if path_valid.is_file():
            data_valid = Dataset.load(context, path_valid)
        else:
            data_valid = Dataset(context)

        # checks; optional
        path_checks = path / 'checks'
        if path_checks.is_dir():
            checks = load_checks(path_checks, context)
        return model, ensemble, data_train, data_valid, checks


def random_learn(
        manager: Manager,
        model: BaseModel,
        reference: BaseReference,
        walker: RandomWalker,
        checks: Optional[list] = None,
        nstates_train: int = 20,
        nstates_valid: int = 5,
        ):
    nstates = nstates_train + nstates_valid
    ensemble = Ensemble.from_walker(walker, nwalkers=nstates)
    data = ensemble.propagate(nstates, checks=checks)
    data = reference.evaluate(data)
    data_train = data[nstates_train:]
    data_valid = data[:nstates_train]
    model.initialize(data_train)
    model.train(data_train, data_valid)
    model.deploy()
    manager.save(
            prefix='random',
            model=model,
            ensemble=ensemble,
            data_train=data_train,
            data_valid=data_valid,
            )
    return data_train, data_valid
