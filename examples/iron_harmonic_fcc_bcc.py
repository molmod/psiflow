import numpy as np
from ase.build import bulk, make_supercell
from ase.units import kB

import psiflow
from psiflow.data import Dataset
from psiflow.free_energy import Integration
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Harmonic, get_mace_mp0
from psiflow.hamiltonians._harmonic import compute_frequencies
from psiflow.tools import compute_harmonic, optimize


def create_harmonic(geometry, hamiltonian):
    minimum = optimize(geometry, hamiltonian, ftol=1e-4, steps=1000, mode='bfgstrm')
    hessian = compute_harmonic(minimum, hamiltonian, pos_shift=0.001)
    return Harmonic(minimum, hessian)


def compute_delta_f(harmonic: Harmonic, scaling: float, temperature: float):
    integration = Integration(
        harmonic,
        temperatures=[temperature],
        delta_hamiltonian=(scaling - 1) * harmonic,  # makes sampling distribution slightly broader
        delta_coefficients=np.linspace(0, 1, num=10, endpoint=True),
    )
    _ = integration.create_walkers(
            Dataset([harmonic.reference_geometry]),
            timestep=3,
            )  # heavy atoms
    integration.sample(steps=5000, step=10, fix_com=True)
    integration.compute_gradients()

    beta = 1 / (kB * temperature)
    f_delta = integration.along_delta(temperature=temperature).result()
    return f_delta[-1] / beta # at hamiltonian H = 0.9 * harmonic


def main():
    iron = bulk("Fe", "bcc", a=2.87, orthorhombic=True)
    bcc = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))
    iron = bulk("Fe", "fcc", a=3.57, orthorhombic=True)
    fcc = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))

    mace = get_mace_mp0('small')
    scaling = 0.9
    temperature = 800

    harmonic_bcc = create_harmonic(bcc, mace)
    harmonic_fcc = create_harmonic(fcc, mace)

    f_harmonic_bcc = harmonic_bcc.compute_free_energy(temperature=temperature, quantum=False)
    f_harmonic_fcc = harmonic_fcc.compute_free_energy(temperature=temperature, quantum=False)

    delta_f_bcc = compute_delta_f(harmonic_bcc, scaling, temperature)
    delta_f_fcc = compute_delta_f(harmonic_fcc, scaling, temperature)

    # analytic reference
    scaled_harmonic_bcc = Harmonic(harmonic_bcc.reference_geometry, 0.9 * harmonic_bcc.hessian.result())
    scaled_harmonic_fcc = Harmonic(harmonic_fcc.reference_geometry, 0.9 * harmonic_fcc.hessian.result())

    scaled_f_bcc = scaled_harmonic_bcc.compute_free_energy(temperature=temperature, quantum=False)
    scaled_f_fcc = scaled_harmonic_fcc.compute_free_energy(temperature=temperature, quantum=False)

    theoretical_ddf = scaled_f_bcc.result() - scaled_f_fcc.result()
    print('theoretical delta(delta(F)) [eV]: {}'.format(theoretical_ddf))

    computed_ddf = f_harmonic_bcc.result() - f_harmonic_fcc.result()
    computed_ddf += delta_f_bcc - delta_f_fcc
    print('   computed delta(delta(F)) [eV]: {}'.format(computed_ddf))


if __name__ == "__main__":
    with psiflow.load():
        main()
