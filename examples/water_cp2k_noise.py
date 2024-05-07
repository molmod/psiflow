import copy

import numpy as np

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.reference import CP2K


def main():
    geometry = Geometry.from_string(
        """
    18
Lattice="7 0 0 0 7 0 0 0 7"
O          7.5344  -47.1952    0.4765
H          8.4160  -47.2312    0.0120
H          7.2660  -46.2693    0.4279
O          8.1630  -50.9274    0.5473
H          8.2441  -50.4060    1.3688
H          7.3554  -50.5591    0.1389
O          9.8927  -47.4450   -0.7542
H         10.6734  -47.2098   -0.2369
H         10.0133  -48.4102   -0.9487
O         10.1564  -50.1025   -1.0813
H         10.0613  -50.5472   -1.9320
H          9.4474  -50.4883   -0.5041
O          8.2958  -48.9198    2.5834
H          7.7974  -48.8686    3.4078
H         7.9420  -48.2014    2.0231
O          6.1151  -49.3715   -0.6444
H         6.4601  -48.5223   -0.3019
H         5.1679  -49.3538   -0.4625
"""
    )
    with open("data/cp2k_input.txt", "r") as f:
        cp2k_input = f.read()
    cp2k = CP2K(cp2k_input)

    delta_x = 0.001
    states = []
    for i in range(50):
        g = copy.deepcopy(geometry)
        g.per_atom.positions += i * delta_x
        states.append(cp2k.evaluate(g))
    data = Dataset(states)
    energy = data.get("per_atom_energy").result()
    forces = data.get("forces").result()

    e_avg = np.mean(energy)
    f_avg = np.mean(forces, axis=0, keepdims=True)

    e_rmse = np.sqrt(np.mean((energy - e_avg) ** 2))
    f_rmse = np.sqrt(np.mean((forces - f_avg) ** 2))

    print("RMSE(energy) [meV/atom]: {}".format(e_rmse * 1000))
    print("RMSE(forces) [meV/angstrom]: {}".format(f_rmse * 1000))


if __name__ == "__main__":
    with psiflow.load():
        main()
