from typing import Callable

import numpy as np
import typeguard
from ase.data import chemical_symbols
from parsl.app.app import python_app

from psiflow.data import Geometry


class ForceMagnitudeException(Exception):
    pass


def check_forces(
    forces: np.ndarray,
    geometry: Geometry,
    max_force: float,
):
    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if np.sum(exceeded):
        indices = np.arange(len(geometry))[exceeded]
        numbers = geometry.per_atom.numbers[exceeded]
        symbols = [chemical_symbols[n] for n in numbers]
        raise ForceMagnitudeException(
            "\nforce exceeded {} eV/A for atoms {}"
            " with chemical elements {}\n".format(
                max_force,
                indices,
                symbols,
            )
        )
    else:
        pass


@typeguard.typechecked
def evaluate_function(
    load_calculators: Callable,
    inputs: list = [],
    outputs: list = [],
    **parameters,  # dict values can be futures, so app must wait for those
) -> None:
    import numpy as np
    from ase import Atoms

    from psiflow.data.geometry import _read_frames, _write_frames

    assert len(inputs) >= 1
    assert len(outputs) == 1
    states = _read_frames(inputs=[inputs[0]])
    calculators, index_mapping = load_calculators(states, inputs[1], **parameters)
    for i, state in enumerate(states):
        calculator = calculators[index_mapping[i]]
        calculator.reset()
        atoms = Atoms(
            numbers=state.per_atom.numbers,
            positions=state.per_atom.positions,
            cell=state.cell,
            pbc=state.periodic,
        )
        atoms.calc = calculator
        state.energy = atoms.get_potential_energy()
        state.per_atom.forces[:] = atoms.get_forces()
        if state.periodic:
            try:  # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            state.stress = stress
    _write_frames(*states, outputs=[outputs[0]])


@typeguard.typechecked
def _add_contributions(
    coefficients: tuple[float, ...],
    inputs: list = [],
    outputs: list = [],
) -> None:
    import copy

    from psiflow.data.geometry import _read_frames, _write_frames

    contributions = [_read_frames(inputs=[i]) for i in inputs]
    assert len(contributions) == len(coefficients)
    length = len(contributions[0])
    for contribution in contributions:
        assert len(contribution) == length

    data = []
    for i in range(length):
        geometries = [contribution[i] for contribution in contributions]
        energy_list = [geometry.energy for geometry in geometries]
        forces_list = [geometry.per_atom.forces for geometry in geometries]

        energy = sum([energy_list[i] * c for i, c in enumerate(coefficients)])
        forces = sum([forces_list[i] * c for i, c in enumerate(coefficients)])

        geometry = copy.deepcopy(geometries[0])
        geometry.energy = energy
        geometry.per_atom.forces[:] = forces

        if geometry.periodic:
            stress_list = [g.stress for g in geometries]
            stress = sum([stress_list[i] * c for i, c in enumerate(coefficients)])
            geometry.stress = stress
        data.append(geometry)
    _write_frames(*data, outputs=[outputs[0]])


add_contributions = python_app(_add_contributions, executors=["default_threads"])
