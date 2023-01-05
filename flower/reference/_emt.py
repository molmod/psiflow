from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List
import typeguard

from ase import Atoms

from parsl.app.app import python_app

from flower.data import FlowerAtoms
from flower.execution import ReferenceExecutionDefinition, ExecutionContext
from flower.reference.base import BaseReference, EmptyParameters


@typeguard.typechecked
def evaluate_emt(
        atoms: Union[Atoms, FlowerAtoms],
        parameters: EmptyParameters,
        inputs: List = [],
        outputs: List = [],
        ):
    from ase.calculators.emt import EMT
    atoms.calc = EMT()
    atoms.info['energy']   = atoms.get_potential_energy()
    atoms.arrays['forces'] = atoms.get_forces()
    atoms.info['stress']   = atoms.get_stress(voigt=False)
    atoms.calc = None
    atoms.evaluation_log  = ''
    atoms.evaluation_flag = 'success'
    return atoms


@typeguard.typechecked
class EMTReference(BaseReference):
    """Container class for EMT calculations (only used for testing purposes)"""

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        label = context[ReferenceExecutionDefinition].label
        app_evaluate_single = python_app(evaluate_emt, executors=[label])
        context.register_app(cls, 'evaluate_single', app_evaluate_single)
        # see https://stackoverflow.com/questions/1817183/using-super-with-a-class-method
        super(EMTReference, cls).create_apps(context)
