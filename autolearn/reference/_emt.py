from parsl.app.app import python_app

from autolearn.execution import ReferenceExecutionDefinition
from autolearn.reference.base import BaseReference


def evaluate_emt(atoms, parameters, inputs=[], outputs=[]):
    from ase.calculators.emt import EMT
    atoms.calc = EMT()
    atoms.info['energy']   = atoms.get_potential_energy()
    atoms.arrays['forces'] = atoms.get_forces()
    atoms.info['stress']   = atoms.get_stress(voigt=False)
    atoms.calc = None
    return atoms


class EMTReference(BaseReference):
    """Container class for EMT calculations (only used for testing purposes)"""

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ReferenceExecutionDefinition].executor_label
        app_evaluate_single = python_app(evaluate_emt, executors=[executor_label])
        context.register_app(cls, 'evaluate_single', app_evaluate_single)
        # see https://stackoverflow.com/questions/1817183/using-super-with-a-class-method
        super(EMTReference, cls).create_apps(context)
