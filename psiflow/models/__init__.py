from typing import Union
import typeguard
from pathlib import Path

from psiflow.execution import ExecutionContext

from .base import BaseModel
from ._nequip import NequIPModel
from ._mace import MACEModel, MACEConfig


@typeguard.typechecked
def load_model(context: ExecutionContext, path: Union[Path, str]) -> BaseModel:
    from pathlib import Path
    import yaml
    from parsl.data_provider.files import File
    from psiflow.utils import copy_app_future, copy_data_future
    path = Path(path)
    assert path.is_dir()
    classes = [
            NequIPModel,
            MACEModel,
            None,
            ]
    for model_cls in classes:
        assert model_cls is not None
        path_config_raw  = path / (model_cls.__name__ + '.yaml')
        if path_config_raw.is_file():
            break
    with open(path_config_raw, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = model_cls(context, config)
    path_config = path / 'config_after_init.yaml'
    path_model  = path / 'model_undeployed.pth'
    if path_model.is_file():
        assert path_config.is_file()
        with open(path_config, 'r') as f:
            config_init = yaml.load(f, Loader=yaml.FullLoader)
        model.config_future = copy_app_future(config_init)
        model.model_future = copy_data_future(
                inputs=[File(str(path_model))],
                outputs=[context.new_file('model_', '.pth')],
                ).outputs[0]
    return model
