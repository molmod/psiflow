import covalent as ct

from ase.calculators.emt import EMT

from autolearn.base import BaseReference


def get_evaluate_electron(reference_execution):
    def evaluate_barebones(sample, reference):
        atoms = sample.atoms.copy()
        atoms.calc = EMT()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        sample.label(energy, forces, stress, log='')
        sample.tag('success')
        return sample
    return ct.electron(evaluate_barebones, executor=reference_execution.executor)


class EMTReference(BaseReference):
    """Implements an EMT calculator"""

    def evaluate(self, sample, reference_execution):
        def evaluate_barebones(sample, reference):
            atoms = sample.atoms.copy()
            atoms.calc = EMT()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress(voigt=False)
            sample.label(energy, forces, stress, log='')
            sample.tag('success')
            return sample
        evaluate_electron = ct.electron(
                evaluate_barebones,
                executor=reference_execution.executor,
                )
        return evaluate_electron(sample, self)
