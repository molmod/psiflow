from pathlib import Path

from ase.io import read

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.reference import NWChemReference
from psiflow.walkers import PlumedBias, BiasedDynamicWalker, \
        DynamicWalker
from psiflow.models import MACEModel, MACEConfig
from psiflow.metrics import Metrics
from psiflow.learning import SequentialLearning


def get_reference():
    calculator_kwargs = {
            'basis': {e: 'cc-pvdz' for e in ['H', 'C', 'O', 'N']},
            'dft': {
                'xc': 'pw91lda',
                'mult': 1,
                'convergence': {
                    'energy': 1e-6,
                    'density': 1e-6,
                    'gradient': 1e-6,
                    },
                #'disp': {'vdw': 3},
                },
            }
    return NWChemReference(**calculator_kwargs)


def get_bias():
    plumed_input = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
phi: TORSION ATOMS=13,15,17,1 
psi: TORSION ATOMS=15,17,1,3 

METAD  LABEL=bias ARG=phi,psi SIGMA=0.2,0.2 GRID_MIN=-pi,-pi GRID_MAX=pi,pi HEIGHT=0.41 PACE=5 TEMP=300 BIASFACTOR=10
"""
    return PlumedBias(plumed_input)


def main(path_output):
    assert not path_output.exists()
    atoms = FlowAtoms.from_atoms(read('data/alanine.xyz'))
    reference = get_reference()
    bias      = get_bias()

    config = MACEConfig()
    config.r_max = 5.0
    config.hidden_irreps = '16x0e + 16x1o'
    config.MLP_irreps = '8x0e'
    config.batch_size = 1
    config.patience = 4
    model = MACEModel(config)

    model.add_atomic_energy('H', reference.compute_atomic_energy('H'))
    model.add_atomic_energy('C', reference.compute_atomic_energy('C'))
    model.add_atomic_energy('O', reference.compute_atomic_energy('O'))
    model.add_atomic_energy('N', reference.compute_atomic_energy('N'))

    learning = SequentialLearning(
            path_output,
            niterations=10,
            initial_temperature=50,
            final_temperature=600,
            pretraining_nstates = 10,
            pretraining_amplitude_pos = 0.02,
            metrics=Metrics('alanine_nwchem', 'psiflow_examples'),
            error_thresholds_for_reset=(5, 100),
            )

    # construct walkers; mix metadynamics with regular MD
    walkers_md = DynamicWalker.multiply(
            5,
            data_start=Dataset([atoms]),
            timestep=0.5,
            steps=300,
            step=30,
            temperature_reset_quantile=0.10, # reset if P(temp) < 0.1
            )
    walkers_mtd = BiasedDynamicWalker.multiply(
            5,
            data_start=Dataset([atoms]),
            bias=bias,
            timestep=0.5,
            steps=200,
            step=20,
            temperature_reset_quantile=0.10, # reset if P(temp) < 0.1
            )
    data = learning.run(
            model=model,
            reference=reference,
            walkers=walkers_md + walkers_mtd,
            )


if __name__ == '__main__':
    psiflow.load()
    path_output = Path.cwd() / 'output'
    main(path_output)
