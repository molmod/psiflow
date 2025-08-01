# TODO: remove psiflow deps
import json

import numpy as np
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.parallel import world
try:
    from dftd3.ase import DFTD3
    from gpaw import GPAW as GPAWCalculator
except ModuleNotFoundError:
    # module is imported from the main Psiflow process
    pass

from psiflow.geometry import Geometry


def minimal_box(
    atoms: Atoms,
    border: float = 0.0,
    h: float = 0.2,
    multiple: int = 4,
) -> None:
    # inspired by gpaw.cluster.Cluster
    if len(atoms) == 0:
        return None
    min_bounds, max_bounds = np.array(
        [np.minimum.reduce(atoms.positions), np.maximum.reduce(atoms.positions)]
    )
    if isinstance(border, list):
        b = np.array(border)
    else:
        b = np.array([border, border, border])
    if not hasattr(h, "__len__"):
        h = np.array([h, h, h])
    min_bounds -= b
    max_bounds += b - min_bounds
    grid_points = np.ceil(max_bounds / h / multiple) * multiple
    length_diff = grid_points * h - max_bounds
    max_bounds += length_diff
    min_bounds -= length_diff / 2
    shift = tuple(-1.0 * min_bounds)
    atoms.translate(shift)
    atoms.set_cell(tuple(max_bounds))


if __name__ == "__main__":

    with open("input.json", "r") as f:
        input_dict = json.loads(f.read())

    geometry = Geometry.from_string(input_dict["geometry"])
    gpaw_parameters = input_dict["gpaw_parameters"]
    properties = input_dict["properties"]
    d3 = gpaw_parameters.pop("d3", {})

    atoms = Atoms(
        numbers=np.copy(geometry.per_atom.numbers),
        positions=np.copy(geometry.per_atom.positions),
        cell=np.copy(geometry.cell),
        pbc=geometry.periodic,
    )
    if not geometry.periodic:
        minimal_box(
            atoms,
            gpaw_parameters.get("h", 0.2),
            gpaw_parameters.pop("minimal_box_border", 2),  # if present, remove
            gpaw_parameters.pop("minimal_box_multiple", 4),
        )

    calculator = GPAWCalculator(**gpaw_parameters)
    if len(d3) > 0:
        calculator = SumCalculator([calculator, DFTD3(**d3)])
    atoms.calc = calculator

    if "forces" in properties:
        geometry.per_atom.forces[:] = atoms.get_forces()
    if "energy" in properties:
        geometry.energy = atoms.get_potential_energy()

    output_str = geometry.to_string()
    if world.rank == 0:
        print("CALCULATION SUCCESSFUL")
        print(output_str)
