from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import covalent as ct

from autolearn.base import BaseWalker
from .utils import apply_strain


@dataclass
class RandomParameters:
    amplitude_pos: float = 0.05
    amplitude_box: float = 0.05
    seed         : int = 0


class RandomWalker(BaseWalker):

    def __init__(self, sample, **kwargs):
        start = deepcopy(sample)
        start.clear()
        self.start = start
        self.state = deepcopy(start)
        self.parameters = RandomParameters(**kwargs)

    def propagate(self, model, model_execution):
        # model and model_execution are ignored here!
        def propagate_barebones(walker):
            np.random.seed(walker.parameters.seed)
            frac = walker.start.atoms.positions @ np.linalg.inv(walker.start.atoms.cell)
            strain = np.random.uniform(
                    -walker.parameters.amplitude_box,
                    walker.parameters.amplitude_box,
                    size=(3, 3))
            strain[0, 1] = strain[1, 0] # strain is symmetric
            strain[0, 2] = strain[2, 0]
            strain[1, 2] = strain[2, 1]
            box = apply_strain(strain, walker.start.atoms.cell)
            positions = frac @ box
            positions += np.random.uniform(
                    -walker.parameters.amplitude_pos,
                    walker.parameters.amplitude_pos,
                    size=walker.start.atoms.positions.shape,
                    )
            walker.state.atoms.set_positions(positions)
            walker.state.atoms.set_cell(box)
            return walker
        propagate_electron = ct.electron(
                propagate_barebones,
                executor='local',
                )
        return propagate_electron(self)
