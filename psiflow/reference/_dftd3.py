import json

import numpy as np
from ase import Atoms
from dftd3.ase import DFTD3

from psiflow.geometry import Geometry


if __name__ == "__main__":

    with open("input.json", "r") as f:
        input_dict = json.loads(f.read())

    geometry = Geometry.from_string(input_dict["geometry"])
    parameters = input_dict["parameters"]
    properties = input_dict["properties"]

    atoms = Atoms(
        numbers=np.copy(geometry.per_atom.numbers),
        positions=np.copy(geometry.per_atom.positions),
        cell=np.copy(geometry.cell),
        pbc=geometry.periodic,
    )

    calculator = DFTD3(**parameters)
    atoms.calc = calculator

    if "forces" in properties:
        geometry.per_atom.forces[:] = atoms.get_forces()
    if "energy" in properties:
        geometry.energy = atoms.get_potential_energy()

    output_str = geometry.to_string()
    print("CALCULATION SUCCESSFUL")
    print(output_str)
