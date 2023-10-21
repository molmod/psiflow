from __future__ import annotations  # necessary for type-guarding class methods

import os
from pathlib import Path
from typing import Union

import typeguard

from psiflow.data import FlowAtoms
from psiflow.utils import resolve_and_check

from .base import BaseWalker  # import base and bias before walkers
from .bias import PlumedBias
from .dynamic import BiasedDynamicWalker, DynamicWalker
from .optimization import OptimizationWalker
from .random import RandomWalker


@typeguard.typechecked
def load_walker(path: Union[Path, str]) -> BaseWalker:
    from pathlib import Path

    import yaml
    from ase.io import read

    from psiflow.utils import copy_app_future

    path = resolve_and_check(Path(path))
    assert path.is_dir()
    path_state0 = path / "state0.xyz"
    path_state = path / "state.xyz"
    assert path_state0.is_file()
    assert path_state.is_file()
    classes = [
        RandomWalker,
        DynamicWalker,
        OptimizationWalker,
        BiasedDynamicWalker,
        None,
    ]
    for walker_cls in classes:
        assert walker_cls is not None
        path_pars = path / (walker_cls.__name__ + ".yaml")
        if path_pars.is_file():
            break
    state0 = FlowAtoms.from_atoms(read(str(path_state0)))
    state = FlowAtoms.from_atoms(read(str(path_state)))
    with open(path_pars, "r") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
        counter = parameters.pop("counter")
    if walker_cls == BiasedDynamicWalker:
        path_plumed = path / ("plumed_input.txt")
        assert path_plumed.is_file()  # has to exist
        bias = PlumedBias.load(path)
        walker = walker_cls(state0, bias=bias, **parameters)
    else:
        walker = walker_cls(state0, **parameters)
    walker.set_state(state)
    walker.counter = copy_app_future(counter)
    return walker


def load_walkers(
    path: Union[Path, str],
) -> list[BaseWalker]:
    path = resolve_and_check(Path(path))
    assert path.is_dir()
    walkers = []
    for name in os.listdir(path):
        if not (path / name).is_dir():
            continue
        path_walker = path / name
        walker = load_walker(path_walker)
        walkers.append(walker)
    return walkers


def save_walkers(
    walkers: list[BaseWalker],
    path: Union[Path, str],
    require_done: bool = True,
):
    path = resolve_and_check(Path(path))
    assert path.is_dir()
    for i, walker in enumerate(walkers):
        path_walker = path / str(i)
        path_walker.mkdir(parents=False)
        walker.save(path_walker, require_done=require_done)
