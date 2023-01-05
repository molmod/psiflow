from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List
import typeguard
from pathlib import Path

from flower.data import FlowerAtoms
from flower.execution import ExecutionContext

from .base import BaseWalker # import base and bias before walkers
from .bias import PlumedBias
from .random import RandomWalker
from .dynamic import DynamicWalker
from .optimization import OptimizationWalker


@typeguard.typechecked

def load_walker(
        context: ExecutionContext,
        path: Union[Path, str],
        ) -> BaseWalker: # not in BaseWalker to avoid circular import
    from pathlib import Path
    from ase.io import read
    import yaml
    from flower.utils import copy_app_future
    path = Path(path)
    assert path.is_dir()
    path_start = path / 'start.xyz'
    path_state = path / 'state.xyz'
    assert path_start.is_file()
    assert path_state.is_file()
    classes = [
            RandomWalker,
            DynamicWalker,
            OptimizationWalker,
            None,
            ]
    for walker_cls in classes:
        assert walker_cls is not None
        path_pars  = path / (walker_cls.__name__ + '.yaml')
        if path_pars.is_file():
            break
    start = FlowerAtoms.from_atoms(read(str(path_start)))
    state = FlowerAtoms.from_atoms(read(str(path_state)))
    with open(path_pars, 'r') as f:
        pars_dict = yaml.load(f, Loader=yaml.FullLoader)
    walker = walker_cls(context, state, **pars_dict)
    walker.start_future = copy_app_future(start)
    return walker
