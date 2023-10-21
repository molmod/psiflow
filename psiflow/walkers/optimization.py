from __future__ import annotations  # necessary for type-guarding class methods

from collections import namedtuple
from typing import Any, Callable, NamedTuple, Optional, Type, Union

import typeguard
from ase import Atoms
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.executors import WorkQueueExecutor

import psiflow
from psiflow.data import FlowAtoms, NullState
from psiflow.models import BaseModel
from psiflow.utils import get_active_executor, unpack_i
from psiflow.walkers.base import BaseWalker

Metadata = namedtuple("Metadata", ["state", "counter", "reset", "time"])


@typeguard.typechecked
def optimize_geometry(
    device: str,
    ncores: int,
    state: FlowAtoms,
    pars: dict[str, Any],
    load_calculator: Callable,
    keep_trajectory: bool = False,
    plumed_input: str = "",
    inputs: list[File] = [],
    outputs: list[File] = [],
    walltime: float = 1e9,
    parsl_resource_specification: Optional[dict] = None,
) -> tuple[FlowAtoms, int, bool, float]:
    import os
    import tempfile
    import time

    import numpy as np
    import torch
    from ase.constraints import ExpCellFilter
    from ase.io import read
    from ase.io.extxyz import write_extxyz
    from ase.optimize.precon import Exp, PreconLBFGS
    from parsl.app.errors import AppTimeout

    from psiflow.data import FlowAtoms

    if device == "cpu":
        torch.set_num_threads(ncores)
    t0 = time.time()

    np.random.seed(pars["seed"])
    torch.manual_seed(pars["seed"])
    atoms = state.copy()
    atoms.calc = load_calculator(inputs[0].filepath, device)
    preconditioner = Exp(A=3)  # from ASE docs
    if pars["optimize_cell"]:  # include cell DOFs in optimization
        try:  # some models do not have stress support; prevent full cell opt!
            stress = atoms.get_stress()
        except Exception:
            raise ValueError("cell optimization requires stress support in model")
        dof = ExpCellFilter(atoms, mask=[True] * 6)
    else:
        dof = atoms
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
    tmp.close()
    path_traj = tmp.name  # dummy log file
    optimizer = PreconLBFGS(
        dof,
        precon=preconditioner,
        use_armijo=True,
        trajectory=path_traj,
    )
    reset = False
    try:
        optimizer.run(fmax=pars["fmax"])
        nsteps = optimizer.nsteps
    except (AppTimeout, RuntimeError) as error:
        print(error)
        nsteps = optimizer.nsteps
    except:
        reset = True
        nsteps = 0
    atoms.calc = None
    if keep_trajectory:
        assert str(outputs[0].filepath).endswith(".xyz")
        with open(outputs[0], "w") as f:
            trajectory = read(path_traj, index=":")
            write_extxyz(f, trajectory)
    if reset:
        atoms = NullState
    os.unlink(path_traj)
    return FlowAtoms.from_atoms(atoms), optimizer.nsteps, reset, time.time() - t0


@typeguard.typechecked
class OptimizationWalker(BaseWalker):
    def __init__(
        self,
        atoms: Union[Atoms, FlowAtoms, AppFuture],
        optimize_cell: bool = True,
        fmax: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__(atoms, **kwargs)
        self.optimize_cell = optimize_cell
        self.fmax = fmax

    def _propagate(self, model, keep_trajectory, file):
        name = model.__class__.__name__
        context = psiflow.context()
        try:
            app = context.apps(OptimizationWalker, "propagate_" + name)
        except KeyError:
            self.create_apps(model_cls=model.__class__)
            app = context.apps(OptimizationWalker, "propagate_" + name)
        return app(
            self.state,
            self.parameters,
            model,
            keep_trajectory,
            file,
        )

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters["optimize_cell"] = self.optimize_cell
        parameters["fmax"] = self.fmax
        return parameters

    @classmethod
    def create_apps(cls, model_cls: Type[BaseModel]) -> None:
        context = psiflow.context()
        evaluation, _ = context[model_cls]
        ncores = evaluation.cores_per_worker
        device = "cuda" if evaluation.gpu else "cpu"
        label = evaluation.name()
        walltime = evaluation.max_walltime
        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_spec = evaluation.generate_parsl_resource_specification()
        else:
            resource_spec = {}

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
        ) -> tuple[NamedTuple, Optional[DataFuture]]:
            assert model is not None  # model is required
            inputs = [model.deploy_future]
            outputs = []
            if keep_trajectory:
                assert file is not None
                outputs.append(file)
            result = app_optimize(
                device,
                ncores,
                state,
                parameters,
                model.load_calculator,  # load function
                keep_trajectory=keep_trajectory,
                inputs=inputs,
                outputs=outputs,
                walltime=(60 * walltime),
                parsl_resource_specification=resource_spec,
            )
            if keep_trajectory:
                output_future = result.outputs[0]
            else:
                output_future = None
            metadata = Metadata(*[unpack_i(result, i) for i in range(4)])
            return metadata, output_future

        name = model_cls.__name__
        context.register_app(cls, "propagate_" + name, optimize_wrapped)
        super(OptimizationWalker, cls).create_apps()
