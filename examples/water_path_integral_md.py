import numpy as np
from ase.neighborlist import primitive_neighbor_list

import psiflow
from psiflow.data import Dataset
from psiflow.hamiltonians import get_mace_mp0
from psiflow.sampling import Walker, sample


def get_OH_distances(geometries):
    distances = []
    for geometry in geometries:
        d = primitive_neighbor_list(
            "d",
            3 * [True],
            geometry.cell,
            geometry.per_atom.positions,
            cutoff={("O", "H"): 1.5},
            numbers=geometry.per_atom.numbers,
        )
        distances.append(d)
    min_n_distances = min([len(d) for d in distances])
    return np.array([d[:min_n_distances] for d in distances]).flatten()


def main():
    geometry = Dataset.load("data/h2o_32.xyz")[0]
    mace = get_mace_mp0()

    trajectories = []
    for i in range(6):
        walker = Walker(
            geometry,
            mace,
            nbeads=2**i,
            temperature=300,
        )
        outputs = sample([walker], steps=2000, step=20)
        trajectories.append(outputs[0].trajectory)

    for i, trajectory in enumerate(trajectories):
        distances = get_OH_distances(trajectory.geometries().result())
        nbeads = 2**i
        std = np.std(distances)
        print("nbeads = {} --> std(O-H) = {} A".format(nbeads, std))


if __name__ == "__main__":
    with psiflow.load():
        main()
