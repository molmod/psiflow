from typing import Callable, Union

import numpy as np
import typeguard
from ase import Atoms
from parsl.app.app import python_app

from psiflow.data import FlowAtoms


class ForceMagnitudeException(Exception):
    pass


def check_forces(
    forces: np.ndarray,
    atoms: Union[Atoms, FlowAtoms],
    max_force: float,
):
    exceeded = np.linalg.norm(forces, axis=1) > max_force
    if np.sum(exceeded):
        indices = np.arange(len(atoms))[exceeded]
        symbols = np.array(atoms.symbols)[exceeded]
        raise ForceMagnitudeException(
            "force exceeded {} eV/A for atoms {}"
            " with chemical elements {}".format(
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

    from psiflow.data import read_dataset, write_dataset

    assert len(inputs) >= 1
    assert len(outputs) == 1
    data = read_dataset(slice(None), inputs=[inputs[0]])
    calculators, index_mapping = load_calculators(data, *inputs[1:], **parameters)
    for i, atoms in enumerate(data):
        calculator = calculators[index_mapping[i]]
        calculator.reset()
        atoms.reset()
        atoms.calc = calculator
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        if atoms.pbc.any():
            try:  # some models do not have stress support
                stress = atoms.get_stress(voigt=False)
            except Exception as e:
                print(e)
                stress = np.zeros((3, 3))
            atoms.info["stress"] = stress
        else:  # remove if present
            atoms.info.pop("stress", None)
        atoms.calc = None
    write_dataset(data, outputs=[outputs[0]])


@typeguard.typechecked
def add_contributions(
    coefficients: tuple[float, ...],
    inputs: list = [],
    outputs: list = [],
) -> None:
    from psiflow.data import read_dataset, write_dataset

    contributions = [read_dataset(slice(None), inputs=[i]) for i in inputs]
    assert len(contributions) == len(coefficients)
    length = len(contributions[0])
    for contribution in contributions:
        assert len(contribution) == length

    data = []
    for i in range(length):
        atoms_list = [contribution[i] for contribution in contributions]
        energy_list = [atoms.info["energy"] for atoms in atoms_list]
        forces_list = [atoms.arrays["forces"] for atoms in atoms_list]

        energy = sum([energy_list[i] * c for i, c in enumerate(coefficients)])
        forces = sum([forces_list[i] * c for i, c in enumerate(coefficients)])
        atoms = atoms_list[0].copy()
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = forces

        if atoms_list[0].pbc.any():
            stress_list = [atoms.info["stress"] for atoms in atoms_list]
            stress = sum([stress_list[i] * c for i, c in enumerate(coefficients)])
            atoms.info["stress"] = stress
        data.append(atoms)
    write_dataset(data, outputs=[outputs[0]])


app_add_contributions = python_app(add_contributions, executors=["default_threads"])
