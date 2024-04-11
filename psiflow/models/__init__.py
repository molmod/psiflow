from pathlib import Path
from typing import Union

import typeguard
import yaml
from ase.data import chemical_symbols
from parsl.data_provider.files import File

import psiflow
from psiflow.models._mace import MACE, MACEConfig  # noqa: F401
from psiflow.models.model import Model
from psiflow.utils import copy_data_future, resolve_and_check


@typeguard.typechecked
def load_model(path: Union[Path, str]) -> Model:
    path = resolve_and_check(Path(path))
    assert path.is_dir()
    classes = [
        MACE,
    ]
    for model_cls in classes + [None]:
        assert model_cls is not None
        name = model_cls.__name__
        path_config = path / (name + ".yaml")
        if path_config.is_file():
            break
    with open(path_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    atomic_energies = {}
    for key in list(config):
        if key.startswith("atomic_energies_"):
            element = key.split("atomic_energies_")[-1]
            assert element in chemical_symbols
            atomic_energies[element] = config.pop(key)
    model = model_cls(**config)
    for element, energy in atomic_energies.items():
        model.add_atomic_energy(element, energy)
    path_model = path / "{}.pth".format(name)
    if path_model.is_file():
        model.model_future = copy_data_future(
            inputs=[File(str(path_model))],
            outputs=[psiflow.context().new_file("model_", ".pth")],
        ).outputs[0]
    return model
