import numpy as np
from ase.units import _c, second

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians import get_mace_mp0
from psiflow.hamiltonians._harmonic import compute_frequencies
from psiflow.sampling import Walker, sample
from psiflow.tools import compute_harmonic, optimize


def frequency_dynamic(start, hamiltonian):
    walker = Walker(
        start,
        hamiltonian=hamiltonian,
        temperature=None,  # NVE!
        timestep=0.25,
    )

    step = 10
    output = sample(
        [walker],
        steps=2000,
        step=step,
        max_force=10,
    )[0]
    positions = output.trajectory.get("positions").result()
    distances = np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=1)
    distances -= np.mean(distances)  # don't need average interatomic distance

    timestep = walker.timestep * 1e-15 * step
    spectrum = np.abs(np.fft.fft(distances))

    freq_axis = np.fft.fftfreq(len(distances), timestep)
    index = np.argmax(spectrum[np.where(freq_axis > 0)])
    peak_frequency = freq_axis[np.where(freq_axis > 0)][index]

    return peak_frequency / (100 * _c)


def frequency_static(start, hamiltonian):
    minimum = optimize(
        start,
        hamiltonian,
        2000,
        ftol=1e-4,
    )
    hessian = compute_harmonic(
        minimum,
        hamiltonian,
        asr="crystal",
        pos_shift=0.001,
    )
    frequencies = compute_frequencies(hessian, minimum).result()
    return frequencies[-1] * second / (100 * _c)


def main():
    geometry = Geometry.from_data(
        numbers=np.ones(2),
        positions=np.array([[0, 0, 0], [0.8, 0, 0]]),
        cell=None,
    )
    mace = get_mace_mp0()

    dynamic = frequency_dynamic(geometry, mace)
    static = frequency_static(geometry, mace)

    print("H2 frequency (dynamic) [inv(cm)]: {}".format(dynamic))
    print("H2 frequency (static)  [inv(cm)]: {}".format(static))


if __name__ == "__main__":
    with psiflow.load():
        main()
