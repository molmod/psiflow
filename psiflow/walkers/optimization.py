from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Callable, Type, Any, Union
import typeguard
from dataclasses import dataclass

from ase import Atoms

from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.app.futures import DataFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.execution import ModelEvaluationExecution
from psiflow.walkers.base import BaseWalker
from psiflow.models import BaseModel


@typeguard.typechecked
def optimize_geometry(
        device: str,
        ncores: int,
        state: FlowAtoms,
        pars: dict[str, Any],
        load_calculator: Callable,
        keep_trajectory: bool = False,
        plumed_input: str = '',
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> tuple[FlowAtoms, str, int]:
    import os
    import tempfile
    import torch
    import numpy as np
    from ase.optimize.precon import Exp, PreconLBFGS
    from ase.constraints import ExpCellFilter
    from ase.io import read
    from ase.io.extxyz import write_extxyz
    from psiflow.data import FlowAtoms
    torch.set_default_dtype(torch.float64) # optimization always in double
    if device == 'cpu':
        torch.set_num_threads(ncores)

    np.random.seed(pars['seed'])
    torch.manual_seed(pars['seed'])
    atoms = state.copy()
    atoms.calc = load_calculator(inputs[0].filepath, device, dtype='float64')
    preconditioner = Exp(A=3) # from ASE docs
    if pars['optimize_cell']: # include cell DOFs in optimization 
        try: # some models do not have stress support; prevent full cell opt!
            stress = atoms.get_stress()
        except Exception as e:
            raise ValueError('cell optimization requires stress support in model')
        dof = ExpCellFilter(atoms, mask=[True] * 6)
    else:
        dof = atoms
    #optimizer = SciPyFminCG(
    #        dof,
    #        trajectory=str(path_traj),
    #        )
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    path_traj = tmp.name # dummy log file
    optimizer = PreconLBFGS(
            dof,
            precon=preconditioner,
            use_armijo=True,
            trajectory=path_traj,
            )

    try:
        optimizer.run(fmax=pars['fmax'])
        nsteps = optimizer.nsteps
    except:
        nsteps = 0
        pass
    atoms.calc = None
    if keep_trajectory:
        assert str(outputs[0].filepath).endswith('.xyz')
        with open(outputs[0], 'w') as f:
            trajectory = read(path_traj, index=':')
            write_extxyz(f, trajectory)
    os.unlink(path_traj)
    return FlowAtoms.from_atoms(atoms), optimizer.nsteps


@typeguard.typechecked
class OptimizationWalker(BaseWalker):

    def __init__(
            self,
            atoms: Union[Atoms, FlowAtoms, AppFuture],
            optimize_cell: bool = True,
            fmax: float = 1e-2,
            **kwargs) -> None:
        super().__init__(atoms, **kwargs)
        self.optimize_cell = optimize_cell
        self.fmax = fmax

    def _propagate(self, model, keep_trajectory, file):
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(OptimizationWalker, 'propagate_' + name)
        except KeyError:
            assert model.__class__ in context.definitions.keys()
            self.create_apps(model_cls=model.__class__)
            app = context.apps(OptimizationWalker, 'propagate_' + name)
        return app(
                self.state_future,
                self.parameters,
                model,
                keep_trajectory,
                file,
                )
    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters['optimize_cell'] = self.optimize_cell
        parameters['fmax'] = self.fmax
        return parameters

    @classmethod
    def create_apps(cls, model_cls: Type[BaseModel]) -> None:
        context = psiflow.context()
        for execution in context[model_cls]:
            if type(execution) == ModelEvaluationExecution:
                label    = execution.executor
                device   = execution.device
                ncores   = execution.ncores

        app_optimize = python_app(
                optimize_geometry,
                executors=[label],
                )
        @typeguard.typechecked
        def optimize_wrapped(
                state: AppFuture,
                parameters: dict[str, Any],
                model: BaseModel = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                **kwargs,
                ) -> tuple[AppFuture, Optional[DataFuture]]:
            assert model is not None # model is required
            assert 'float64' in model.deploy_future.keys() # has to be deployed
            inputs = [model.deploy_future['float64']]
            outputs = []
            if keep_trajectory:
                assert file is not None
                outputs.append(file)
            result = app_optimize(
                    device,
                    ncores,
                    state,
                    parameters,
                    model.load_calculator, # load function
                    keep_trajectory=keep_trajectory,
                    inputs=inputs,
                    outputs=outputs,
                    )
            if keep_trajectory:
                output_future = result.outputs[0]
            else:
                output_future = None
            return result, output_future
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, optimize_wrapped)
        super(OptimizationWalker, cls).create_apps()
