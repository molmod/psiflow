from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Tuple, List, Callable, Type
import typeguard
from dataclasses import dataclass

from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.execution import ModelEvaluationExecution
from psiflow.sampling.base import BaseWalker
from psiflow.models import BaseModel


@typeguard.typechecked
def optimize_geometry(
        device: str,
        ncores: int,
        state: FlowAtoms,
        parameters: OptimizationParameters,
        load_calculator: Callable,
        keep_trajectory: bool = False,
        plumed_input: str = '',
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> Tuple[FlowAtoms, str, int]:
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

    pars = parameters
    np.random.seed(pars.seed)
    torch.manual_seed(pars.seed)
    atoms = state.copy()
    atoms.calc = load_calculator(inputs[0].filepath, device, dtype='float64')
    preconditioner = Exp(A=3) # from ASE docs
    if parameters.optimize_cell: # include cell DOFs in optimization 
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

    tag = 'safe'
    try:
        optimizer.run(fmax=parameters.fmax)
    except:
        tag = 'unsafe'
        pass
    atoms.calc = None
    if keep_trajectory:
        assert str(outputs[0].filepath).endswith('.xyz')
        with open(outputs[0], 'w') as f:
            trajectory = read(path_traj)
            write_extxyz(f, trajectory)
    os.unlink(path_traj)
    return FlowAtoms.from_atoms(atoms), tag, optimizer.nsteps


@typeguard.typechecked
@dataclass
class OptimizationParameters:
    optimize_cell: bool = True # include cell DOFs in optimization
    fmax         : float = 1e-2 # max residual norm of forces before termination
    seed         : int = 0 # seed for randomized initializations


@typeguard.typechecked
class OptimizationWalker(BaseWalker):
    parameters_cls = OptimizationParameters

    def get_propagate_app(self, model):
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(OptimizationWalker, 'propagate_' + name)
        except KeyError:
            assert model.__class__ in context.definitions.keys()
            self.create_apps(model_cls=model.__class__)
            app = context.apps(OptimizationWalker, 'propagate_' + name)
        return app

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
                parameters: OptimizationParameters,
                model: BaseModel = None,
                keep_trajectory: bool = False,
                file: Optional[File] = None,
                **kwargs,
                ) -> AppFuture:
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
            return result
        name = model_cls.__name__
        context.register_app(cls, 'propagate_' + name, optimize_wrapped)
        super(OptimizationWalker, cls).create_apps()
