import json
import io

import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.parallel import world

FILEPATH = __file__
DEFAULTS = dict(d3={}, h=0.2, minimal_box_border=2, minimal_box_multiple=4)
STDOUT_KEY = "CALCULATION SUCCESSFUL"


def minimal_box(
    atoms: Atoms,
    h: float,
    border: float,
    multiple: int,
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
    from dftd3.ase import DFTD3
    from gpaw import GPAW as GPAWCalculator

    with open("input.json", "r") as f:
        input_dict = json.loads(f.read())

    atoms_str = io.StringIO(input_dict["geometry"])
    atoms = ase.io.read(atoms_str, format="extxyz")

    gpaw_parameters = input_dict["gpaw_parameters"]
    if not all(atoms.pbc):
        minimal_box(
            atoms,
            gpaw_parameters.get("h", 0.2),
            gpaw_parameters.pop("minimal_box_border", 2),  # if present, remove
            gpaw_parameters.pop("minimal_box_multiple", 4),
        )

    d3 = gpaw_parameters.pop("d3", {})
    calculator = GPAWCalculator(**gpaw_parameters)
    if len(d3) > 0:
        calculator = SumCalculator([calculator, DFTD3(**d3)])

    atoms.calc = calculator
    if "forces" in input_dict["properties"]:
        atoms.get_forces()
    atoms.get_potential_energy()

    if world.rank == 0:
        print(STDOUT_KEY)
        ase.io.write("-", atoms, format="extxyz")
        print(STDOUT_KEY)
