import numpy as np

from autolearn.reference import EMTReference
from autolearn import ReferenceExecution
from autolearn.utils import clear_label

from utils import generate_emt_cu_data


def test_reference_emt(tmp_path):
    atoms_list = generate_emt_cu_data(a=3.6, nstates=1)
    atoms = atoms_list[0]
    e0 = atoms.info['energy']

    reference = EMTReference() # redo computation via EMTReference
    reference_execution = ReferenceExecution()
    clear_label(atoms)
    assert atoms.info['energy'] == 0.0
    atoms = EMTReference.evaluate(atoms, reference, reference_execution)
    assert np.allclose(e0, atoms.info['energy'])
    assert atoms.calc is None
