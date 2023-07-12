from typing import Union
import typeguard
from pathlib import Path

import psiflow
from psiflow.utils import resolve_and_check

from .base import BaseModel
from ._nequip import NequIPModel, NequIPConfig, AllegroModel, AllegroConfig
from ._mace import MACEModel, MACEConfig
from .committee import CommitteeMixin


class MACECommittee(MACEModel, CommitteeMixin):
    pass


class NequIPCommittee(NequIPModel, CommitteeMixin):
    pass


class AllegroCommittee(AllegroModel, CommitteeMixin):
    pass


@typeguard.typechecked
def load_model(path: Union[Path, str]) -> BaseModel:
    from pathlib import Path
    import yaml
    from parsl.data_provider.files import File
    from psiflow.utils import copy_app_future, copy_data_future
    path = resolve_and_check(Path(path))
    assert path.is_dir()
    classes = [
            NequIPModel,
            AllegroModel,
            MACEModel,
            None,
            ]
    classes_committee = [
            NequIPCommittee,
            MACECommittee,
            AllegroCommittee,
            ]
    for model_cls in classes:
        assert model_cls is not None
        path_config_raw  = path / (model_cls.__name__ + '.yaml')
        if path_config_raw.is_file():
            break
    with open(path_config_raw, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    is_committee = False
    for committee_cls in classes:
        path_config_raw  = path / (model_cls.__name__ + '.yaml')
        if path_config_raw.is_file():
            is_committee = True
            break
    if is_committee:
        model_cls = committee_cls
        with open(path_config_raw, 'r') as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
    else:
        kwargs = {}
    model = model_cls(config, **kwargs)
    path_config = path / 'config_after_init.yaml'
    path_model  = path / 'model_undeployed.pth'
    if path_model.is_file():
        assert path_config.is_file()
        with open(path_config, 'r') as f:
            config_init = yaml.load(f, Loader=yaml.FullLoader)
        model.config_future = copy_app_future(config_init)
        model.model_future = copy_data_future(
                inputs=[File(str(path_model))],
                outputs=[psiflow.context().new_file('model_', '.pth')],
                ).outputs[0]
    if is_committee:
        for i in range(len(model.models)):
            path_model = path / str(i + 1)
            assert path_model.exists()
            model.models[i] = load_model(path_model)
    return model
