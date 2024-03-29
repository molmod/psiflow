from pathlib import Path
from typing import Union

import typeguard
from ase.data import chemical_symbols

import psiflow
from psiflow.utils import resolve_and_check

from ._mace import MACEConfig, MACEModel  # noqa: F401
from ._nequip import (  # noqa: F401
    AllegroConfig,
    AllegroModel,
    NequIPConfig,
    NequIPModel,
)
from .base import BaseModel


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
    for model_cls in classes:
        assert model_cls is not None
        path_config_raw = path / (model_cls.__name__ + ".yaml")
        if path_config_raw.is_file():
            break
    with open(path_config_raw, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = model_cls(config)
    for element in chemical_symbols:
        energy = model.config_raw.pop("atomic_energies_" + element, None)
        if energy is not None:
            model.add_atomic_energy(element, energy)
    path_config = path / "config_after_init.yaml"
    path_model = path / "model_undeployed.pth"
    if path_model.is_file():
        assert path_config.is_file()
        with open(path_config, "r") as f:
            config_init = yaml.load(f, Loader=yaml.FullLoader)
        model.config_future = copy_app_future(config_init)
        model.model_future = copy_data_future(
            inputs=[File(str(path_model))],
            outputs=[psiflow.context().new_file("model_", ".pth")],
        ).outputs[0]
        model.deploy()
    return model
