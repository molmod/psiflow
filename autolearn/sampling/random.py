from dataclasses import dataclass

from parsl.app.app import python_app

from autolearn.sampling import BaseWalker


def random_perturbation(state, parameters):
    import numpy as np
    from autolearn.sampling.utils import apply_strain
    np.random.seed(parameters.seed)
    frac = state.positions @ np.linalg.inv(state.cell)
    strain = np.random.uniform(
            -parameters.amplitude_box,
            parameters.amplitude_box,
            size=(3, 3),
            )
    strain[0, 1] = strain[1, 0] # strain is symmetric
    strain[0, 2] = strain[2, 0]
    strain[1, 2] = strain[2, 1]
    box = apply_strain(strain, state.cell)
    positions = frac @ box
    positions += np.random.uniform(
            -parameters.amplitude_pos,
            parameters.amplitude_pos,
            size=state.positions.shape,
            )
    state.set_positions(positions)
    state.set_cell(box)
    return state


@dataclass
class RandomParameters:
    amplitude_pos: float = 0.05
    amplitude_box: float = 0.05
    seed         : int = 0


class RandomWalker(BaseWalker):
    parameters_cls = RandomParameters

    def propagate(self, model=None):
        p_random_perturbation = python_app(
                random_perturbation,
                executors=[self.executor_label],
                )
        self.state = p_random_perturbation(
                self.state,
                self.parameters,
                )
        self.tag = 'safe' # random perturbation always safe
        return self.state
