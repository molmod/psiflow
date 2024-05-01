import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.hamiltonians import PlumedHamiltonian, get_mace_cc
from psiflow.sampling import HamiltonianOrderParameter
from psiflow.tools import optimize


def main():
    geometry = Geometry.from_string(
        """
10
pbc="F F F"
O        1.04654576      -1.55962797       1.01544860
C        0.14921681      -2.30925077       0.71774831
O       -0.85162780      -2.04467245      -0.08370986
H        0.08320651      -3.33257292       1.10876381
H       -0.77498652      -1.13207313      -0.42019152
O       -0.58268825       0.61409530      -1.03260613
C        0.31461602       1.36373408      -0.73487046
O        1.31543668       1.09916662       0.06662335
H        0.38061481       2.38706612      -1.12587079
H        1.23881398       0.18655913       0.40308220
""",
        None,
    )
    mace = get_mace_cc()

    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

C1: COORDINATION GROUPA=1 GROUPB=10 R_0=1.4
C2: COORDINATION GROUPA=8 GROUPB=10 R_0=1.4
C3: COORDINATION GROUPA=3 GROUPB=5 R_0=1.4
C4: COORDINATION GROUPA=6 GROUPB=5 R_0=1.4

CV: COMBINE ARG=C1,C2,C3,C4 COEFFICIENTS=1.0,-1.0,-1.0,1.0 PERIODIC=NO
RESTRAINT ARG=CV AT=0.0 KAPPA=300
"""
    umbrella = PlumedHamiltonian(plumed_input)
    order = HamiltonianOrderParameter.from_plumed(
        name="CV",
        hamiltonian=umbrella,
    )

    geometries = []
    nsteps = 6
    for i in range(nsteps):
        hamiltonian = mace + (i / nsteps) * umbrella
        final = optimize(
            geometry,
            hamiltonian,
            steps=2000,
        )
        geometries.append(order.evaluate(final))

    CV = Dataset(geometries).get("CV").result()
    print("CV values: ")
    for i in range(nsteps):
        print("\t{}: {}".format(i, CV[i]))


if __name__ == "__main__":
    psiflow.load()
    main()
    psiflow.wait()
