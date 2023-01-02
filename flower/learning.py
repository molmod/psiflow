from dataclasses import dataclass
from typing import Optional

from flower.manager import Manager
from flower.models import BaseModel
from flower.reference import BaseReference
from floer.sampling import RandomWalker
from flower.ensemble import Ensemble


@python_app(executors=['default'])
def get_ntrain_nvalid(effective_nstates, train_valid_split):
    import numpy as np
    ntrain = int(np.floor(nstates * train_valid_split))
    nvalid = nstates - ntrain
    assert ntrain > 0
    assert nvalid > 0
    return ntrain, nvalid


class BaseLearning:
    parameter_cls = None

    def __init__(self, **kwargs):
        self.parameters = parameter_cls(**kwargs)


@dataclass
class RandomLearningParameters:
    nstates           : int = 20
    train_valid_split : int = 0.9


class RandomLearning:
    parameter_cls = RandomLearningParameters

    def run(
            self,
            manager: Manager,
            model: BaseModel,
            reference: BaseReference,
            walker: RandomWalker,
            checks: Optional[list] = None,
            ):
        ensemble = Ensemble.from_walker(walker, nwalkers=self.parameters.nstates)
        data = ensemble.sample(self.parameters.nstates, checks=checks)
        data = reference.evaluate(data)
        data_success = data.get(indices=data.success)
        ntrain, nvalid = get_ntrain_nvalid(
                data_success.length(), # can be less than nstates
                self.parameters.train_valid_split,
                )
        data_train = data_success[ntrain:]
        data_valid = data_success[:ntrain]
        model.initialize(data_train)
        model.train(data_train, data_valid)
        manager.save(
                prefix='random',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                )
        manager.log_dataset( # log training
                wandb_name='train',
                wandb_group='random_learning',
                dataset=data_train,
                visualize_structures=False,
                bias=None,
                model=model,
                )
        return data_train, data_valid


@dataclass
class OnlineLearningParameters(RandomLearningParameters):
    niterations : int = 10
    retrain_model_per_iteration : bool = True


class OnlineLearning:
    parameter_cls = OnlineLearningParameters

    def run(
            self,
            manager: Manager,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble,
            checks: Optional[list] = None,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            ):
        if data_train is None:
            data_train = Dataset(model.context, atoms_list=[])
        if data_valid is None:
            data_valid = Dataset(model.context, atoms_list=[])
        for i in range(self.parameters.niterations):
            model.deploy()
            dataset = ensemble.sample(
                    self.parameters.nstates,
                    model=model,
                    checks=checks,
                    )
            data = reference.evaluate(data)
            data_success = data.get(indices=data.success)
            ntrain, nvalid = get_ntrain_nvalid(
                    data_success.length(), # can be less than nstates
                    self.parameters.train_valid_split,
                    )
            data_train.append(data_success[ntrain:])
            data_valid.append(data_success[:ntrain])
            if self.parameters.retrain_model_per_iteration:
                model.initialize(total_train)
            model.train(data_train, data_valid)
            manager.save(
                    prefix=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    )
