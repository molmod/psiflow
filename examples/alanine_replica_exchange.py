import numpy as np

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians import MACEHamiltonian
from psiflow.sampling import Walker, sample, replica_exchange


def compute_dihedrals(positions):
    indices_phi = np.array([4, 6, 8, 14], dtype=int)
    indices_psi = np.array([6, 8, 14, 16], dtype=int)

    dihedrals = []
    for indices in [indices_phi, indices_psi]:
        p1 = positions[:, indices[0], :]
        p2 = positions[:, indices[1], :]
        p3 = positions[:, indices[2], :]
        p4 = positions[:, indices[3], :]

        # Calculate vectors between the points
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3

        # Normal vectors of the planes formed by the atoms
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)

        # Normalize the normal vectors
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        dot_product = np.einsum("ij,ij->i", n1, n2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        dihedrals.append(np.arccos(dot_product))
    return dihedrals[0], dihedrals[1]  # phi, psi


def main():
    c7eq = np.array([2.8, 2.9])  # noqa: F841
    c7ax = np.array([1.2, -0.9])  # noqa: F841
    alanine = Geometry.from_string(  # starts in c7ax config
        """
22
Properties=species:S:1:pos:R:3 pbc="F F F"
H       12.16254811      17.00740464      -2.89412387
C       12.83019906      16.90038734      -2.04015291
H       12.24899130      16.91941920      -1.11925017
H       13.51243976      17.75054269      -2.01566384
C       13.65038992      15.63877411      -2.06030255
O       14.36738511      15.33906728      -1.11622456
N       13.53865222      14.88589532      -3.17304444
H       12.86898792      15.18433500      -3.85740375
C       14.28974353      13.67606132      -3.48863158
H       14.01914560      13.42643243      -4.51320992
C       15.79729109      13.88220294      -3.42319959
H       16.12104919      14.14072623      -2.41784410
H       16.29775468      12.96420765      -3.73059171
H       16.09643748      14.68243453      -4.10096574
C       13.86282687      12.43546588      -2.69127862
O       13.58257313      11.40703144      -3.28015921
N       13.87365846      12.57688288      -1.35546630
H       14.15017274      13.47981654      -0.98516877
C       13.53768820      11.50108113      -0.46287859
H       14.38392004      11.24258036       0.17699860
H       12.69022125      11.76658121       0.17241519
H       13.27142638      10.63298597      -1.06170510
"""
    )
    mace = MACEHamiltonian.mace_mp0()

    walkers = []
    for temperature in [150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]:
        walker = Walker(
            alanine,
            mace,
            temperature=temperature,
        )
        walkers.append(walker)
    replica_exchange(walkers, trial_frequency=50)

    outputs = sample(walkers, steps=20000, step=200)
    phi, psi = compute_dihedrals(outputs[0].trajectory.get("positions").result())
    for f, s in zip(phi, psi):  # some c7eq conformations should appear here
        print("{:5.3f}  {:5.3f}".format(f, s))


if __name__ == "__main__":
    with psiflow.load():
        main()
