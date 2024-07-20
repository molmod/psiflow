import numpy as np
from ase.build import bulk, make_supercell
from ase.units import kB

import psiflow
from psiflow.data import Dataset
from psiflow.free_energy import Integration, compute_harmonic, harmonic_free_energy
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Harmonic, MACEHamiltonian
from psiflow.sampling import optimize


def main():
    iron = bulk("Fe", "bcc", a=2.87, orthorhombic=True)
    bcc = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))
    iron = bulk("Fe", "fcc", a=3.57, orthorhombic=True)
    fcc = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))

    geometries = {
        "bcc": bcc,
        "fcc": fcc,
    }
    theoretical = {name: None for name in geometries}
    simulated = {name: None for name in geometries}

    mace = MACEHamiltonian.mace_mp0("small")
    scaling = 0.9
    temperature = 800
    beta = 1 / (kB * temperature)

    for name in geometries:
        minimum = optimize(
            geometries[name], mace, ftol=1e-4, steps=1000, mode="bfgstrm"
        )
        hessian = compute_harmonic(minimum, mace, pos_shift=0.001)

        # simulate
        harmonic = Harmonic(minimum, hessian)
        integration = Integration(
            harmonic,
            temperatures=[temperature],
            delta_hamiltonian=(scaling - 1) * harmonic,
            delta_coefficients=np.linspace(0, 1, num=4, endpoint=True),
        )
        walkers = integration.create_walkers(  # noqa: F841
            Dataset([harmonic.reference_geometry]),
            timestep=3,
        )  # heavy atoms
        integration.sample(steps=500, step=10, start=300)
        integration.compute_gradients()

        reduced_f = integration.along_delta(temperature=temperature).result()
        f_harmonic = harmonic_free_energy(
            hessian,
            temperature=temperature,
            quantum=False,
        )
        simulated[name] = (f_harmonic.result() + reduced_f[-1]) / beta

        # theoretical
        f_harmonic_scaled = harmonic_free_energy(
            scaling * hessian.result(),
            temperature=temperature,
            quantum=False,
        )
        theoretical[name] = f_harmonic_scaled.result() / beta

    ddF = theoretical["bcc"] - theoretical["fcc"]
    print("theoretical delta(delta(F)) [eV]: {}".format(ddF))

    ddF = simulated["bcc"] - simulated["fcc"]
    print("  simulated delta(delta(F)) [eV]: {}".format(ddF))


if __name__ == "__main__":
    with psiflow.load():
        main()
