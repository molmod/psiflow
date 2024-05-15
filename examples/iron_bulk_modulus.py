import numpy as np
from ase.build import bulk, make_supercell

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians import get_mace_mp0
from psiflow.sampling import Walker, sample


def main():
    iron = bulk("Fe", "bcc", a=2.8)
    geometry = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))
    mace = get_mace_mp0()

    pressures = (-10 + np.arange(5) * 5) * 1e3  # in MPa
    walkers = [Walker(geometry, mace, temperature=300, pressure=p) for p in pressures]

    name = "volume{angstrom3}"
    outputs = sample(walkers, steps=4000, step=50, observables=[name])
    volumes = [np.mean(o[name].result()) for o in outputs]

    p = np.polyfit(volumes, pressures, deg=1)
    volume0 = (-1.0) * p[1] / p[0]
    bulk_modulus = (-1.0) * volume0 * p[0] / 1000  # in GPa
    print("bulk modulus [GPa]: {}".format(bulk_modulus))


if __name__ == "__main__":
    with psiflow.load():
        main()
