import numpy as np
from ase.build import bulk, make_supercell

import psiflow
from psiflow.data import Dataset
from psiflow.free_energy import Integration
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Harmonic, get_mace_mp0
from psiflow.tools import compute_harmonic, optimize


def main():
    iron = bulk("Fe", "bcc", a=2.8)
    geometry = Geometry.from_atoms(make_supercell(iron, 3 * np.eye(3)))
    mace = get_mace_mp0()
    minimum = optimize(
        geometry,
        mace,
        steps=4000,
        ftol=1e-4,
    ).result()
    hessian = compute_harmonic(
        minimum,
        mace,
        asr="crystal",
        pos_shift=0.005,
    ).result()
    minimum.energy = 0.0
    harmonic = Harmonic(minimum, hessian)  # base reference

    # THERMODYNAMIC INTEGRATION
    # hamiltonian_A = harmonic
    # hamiltonian_B = harmonic + delta
    integration = Integration(
        harmonic,
        temperatures=[300],
        delta=0.05 * harmonic,  # makes sampling distribution slightly broader
        npoints=10,
    )
    _ = integration.create_walkers(Dataset([minimum]), timestep=3)  # heavy atoms
    integration.sample(steps=5000)
    integration.compute_gradients()

    xy = np.zeros((len(integration.scales), 2))
    for i, (scale, state) in enumerate(zip(integration.scales, integration.states)):
        xy[i, 0] = scale
        xy[i, 1] = state.gradients["lambda"].result()

    delta_F = np.trapz(xy[:, 1], xy[:, 0])  # TODO: analytical reference
    print('delta F: {}'.format(delta_F))

if __name__ == "__main__":
    psiflow.load()
    main()
    psiflow.wait()
    psiflow.cleanup()
