"""
Structure optimisation through ASE
"""

import os
import json
import warnings
import signal
import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import ase
import ase.io
import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.mixing import LinearCombinationCalculator
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

from psiflow.geometry import Geometry
from psiflow.functions import function_from_json, Function
from psiflow.sampling.utils import TimeoutException, timeout_handler


ALLOWED_MODES: tuple[str, ...] = ("full", "fix_volume", "fix_shape", "fix_cell")
FILE_OUT: str = "out.xyz"
FILE_TRAJ: str = "out.traj"


class FunctionCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, function: Function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def calculate(
        self,
        atoms=None,
        properties=all_properties,
        system_changes=None,
    ):
        super().calculate(atoms, properties, system_changes)
        geometry = Geometry.from_atoms(self.atoms)
        self.results = self.function(geometry)
        self.results["free_energy"] = self.results["energy"]  # required by optimiser


def log_state(atoms: ase.Atoms) -> None:
    """"""

    def make_log(data: list[tuple[str]]):
        """"""
        txt = ["", "Current atoms state:"]
        txt += [f"{_[0]:<15}: {_[1]:<25}[{_[2]}]" for _ in data]
        txt += "End", ""
        print(*txt, sep="\n")

    data = []
    energy, max_force = [np.nan] * 2
    if atoms.calc:
        energy = atoms.get_potential_energy()
        max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
    data.append(("Energy", f"{energy:.2f}", "eV"))
    data.append(("Max. force", f"{max_force:.2E}", "eV/A"))

    if not all(atoms.pbc):
        make_log(data)
        return

    volume, cell = atoms.get_volume(), atoms.get_cell().cellpar().round(3)
    data.append(("Cell volume", f"{atoms.get_volume():.2f}", "A^3"))
    data.append(("Box norms", str(cell[:3])[1:-1], "A"))
    data.append(("Box angles", str(cell[3:])[1:-1], "degrees"))

    make_log(data)
    return


def get_dof_filter(
    atoms: ase.Atoms, mode: str, pressure: float
) -> ase.Atoms | FrechetCellFilter:
    """"""
    if mode == "fix_cell":
        if pressure:
            warnings.warn("Ignoring external pressure..")
        return atoms
    kwargs = {"mask": [True] * 6, "scalar_pressure": pressure}  # enable cell DOFs
    if mode == "fix_shape":
        kwargs["hydrostatic_strain"] = True
    if mode == "fix_volume":
        kwargs["constant_volume"] = True
        if pressure:
            warnings.warn(
                "Ignoring applied pressure during fixed volume optimisation.."
            )
    return FrechetCellFilter(atoms, **kwargs)


def optimise(
    dof: ase.Atoms | FrechetCellFilter,
    f_max: float,
    max_steps: int,
    trajectory: Path | str = None,
    **kwargs,
) -> bool:
    """"""
    steps = max_steps // 2
    opt = BFGS(dof, trajectory=trajectory)

    converged = opt.run(fmax=f_max, steps=steps)
    if converged:
        return True

    print(f"No convergence after {steps} steps. Resetting optimiser..")
    atoms = opt.atoms.atoms if isinstance(opt.atoms, FrechetCellFilter) else opt.atoms
    atoms.rattle(stdev=1e-4)

    opt = BFGS(dof, trajectory=trajectory, append_trajectory=True)
    return opt.run(fmax=f_max, steps=steps)


def main(args: SimpleNamespace):
    """"""
    config = json.load(Path(args.input_config).open("r"))
    config["trajectory"] = FILE_TRAJ if args.output_traj else None

    atoms = ase.io.read(args.start_xyz)
    if not any(atoms.pbc):
        atoms.center(vacuum=0)  # optimiser mysteriously requires a nonzero unit cell
        if config["mode"] != "fix_cell":
            config["mode"] = "fix_cell"
            warnings.warn("Molecular structure is not periodic. Ignoring cell..")

    # construct calculator by combining hamiltonians
    assert args.path_hamiltonian is not None
    print("Making calculator from:", *config["forces"], sep="\n")
    functions = [function_from_json(p) for p in args.path_hamiltonian]
    calc = LinearCombinationCalculator(
        [FunctionCalculator(f) for f in functions],
        [float(h["weight"]) for h in config["forces"]],
    )

    atoms.calc = calc
    dof = get_dof_filter(atoms, config["mode"], config["pressure"])
    log_state(atoms)

    print(f"pid: {os.getpid()}")
    print(f"CPU affinity: {os.sched_getaffinity(os.getpid())}")
    converged = False
    try:
        converged = optimise(dof, **config)
        assert converged
        print("OPTIMISATION SUCCESSFUL")
    except RuntimeError as e:
        print("OPTIMISATION FAILED", e, sep="\n")
    except TimeoutException:
        print("OPTIMISATION TIMEOUT")
    except AssertionError:
        print("OPTIMISATION DID NOT CONVERGE")

    log_state(atoms)
    if not converged:
        # touch output file so Parsl thinks the task completed regardless
        Path(args.output_xyz).touch()
        return

    if not any(atoms.pbc):
        # remove meaningless cell and stress
        atoms.cell = None
        atoms.calc.results.pop("stress", None)
    ase.io.write(FILE_OUT, atoms, format="extxyz")

    shutil.copy(FILE_OUT, args.output_xyz)
    if args.output_traj is not None:
        atoms = [at for at in ase.io.trajectory.Trajectory(FILE_TRAJ)]
        ase.io.write(args.output_traj, atoms, format="extxyz")
    print("FILES MOVED")

    return


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, timeout_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_hamiltonian", action="extend", nargs="*")
    parser.add_argument("--input_config", required=True)
    parser.add_argument("--start_xyz", required=True)
    parser.add_argument("--output_xyz", required=True)
    parser.add_argument("--output_traj")
    args = parser.parse_args()

    time_start = time.time()
    main(args)
    time_stop = time.time()
    print(f"Total elapsed time: {time_stop - time_start: .3f} seconds")
