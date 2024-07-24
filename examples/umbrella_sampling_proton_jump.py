import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians import PlumedHamiltonian


def get_bias(kappa: float, center: float):
    plumed_str = """UNITS LENGTH=A ENERGY=kj/mol
d_C: DISTANCE ATOMS=3,5
d_O: DISTANCE ATOMS=1,5
CV: COMBINE ARG=d_C,d_O COEFFICIENTS=1,-1 PERIODIC=NO
"""
    plumed_str += '\n'
    plumed_str += 'RESTRAINT ARG=CV KAPPA={} AT={}\n'.format(kappa, center)
    return PlumedHamiltonian(plumed_str)


def main():
    aldehyd = Geometry.load('data/acetaldehyde.xyz')
    alcohol = Geometry.load('data/vinyl_alcohol.xyz')

    mace = MACEHamiltonian.mace_cc()


if __name__ == '__main__':
    with psiflow.load() as f:
        main()
