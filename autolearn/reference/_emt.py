import covalent as ct

from ase.calculators.emt import EMT

from autolearn.base import BaseReference


def get_evaluate_electron(reference_execution):
    def evaluate_barebones(atoms, reference):
        atoms.calc = EMT()
        atoms.info['energy']   = atoms.get_potential_energy()
        atoms.arrays['forces'] = atoms.get_forces()
        atoms.info['stress'] = atoms.get_stress(voigt=False)
        atoms.calc = None
        return atoms
    return ct.electron(evaluate_barebones, executor=reference_execution.executor)


class EMTReference(BaseReference):
    """Implements an EMT calculator"""

    @staticmethod
    def evaluate(atoms, reference, reference_execution):
        evaluate_electron = get_evaluate_electron(reference_execution)
        return evaluate_electron(atoms, reference)
