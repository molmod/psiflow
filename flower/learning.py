from dataclasses import dataclass
from typing import Optional

from flower.manager import Manager
from flower.models import BaseModel
from flower.reference import BaseReference
from floer.sampling import RandomWalker
from flower.ensemble import Ensemble


class BaseLearning:
    parameter_cls = None

    def __init__(self, **kwargs):
        self.parameters = parameter_cls(**kwargs)


@dataclass
class RandomLearningParameters
    nstates_train : int = 20
    nstates_valid : int = 5


class RandomLearning:
    parameter_cls = RandomLearningParameters

    def run(
            manager: Manager,
            model: BaseModel,
            reference: BaseReference,
            walker: RandomWalker,
            checks: Optional[list] = None,
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
