import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import typeguard
from ase.calculators.calculator import Calculator, all_changes
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def try_manual_plumed_linking() -> str:
    if "PLUMED_KERNEL" not in os.environ.keys():
        # try linking manually
        if "CONDA_PREFIX" in os.environ.keys():  # for conda environments
            p = "CONDA_PREFIX"
        elif "PREFIX" in os.environ.keys():  # for pip environments
            p = "PREFIX"
        else:
            raise ValueError("failed to set plumed .so kernel")
        path = os.environ[p] + "/lib/libplumedKernel.so"
        if os.path.exists(path):
            os.environ["PLUMED_KERNEL"] = path
            logging.info("plumed kernel manually set at : {}".format(path))
        else:
            raise ValueError("plumed kernel not found at {}".format(path))
    return os.environ["PLUMED_KERNEL"]


@typeguard.typechecked
def remove_comments_printflush(plumed_input: str) -> str:
    new_input = []
    for line in list(plumed_input.split("\n")):
        if line.strip().startswith("#"):
            continue
        if line.strip().startswith("PRINT"):
            continue
        if line.strip().startswith("FLUSH"):
            continue
        new_input.append(line)
    return "\n".join(new_input)


@typeguard.typechecked
def evaluate_plumed(
    atoms: FlowAtoms,
    plumed_input: str,
    *input_files: str,
) -> tuple:
    """Inspired by ASE's PLUMED interface"""
    from psiflow.hamiltonians._plumed import (
        remove_comments_printflush,
        try_manual_plumed_linking,
    )

    try_manual_plumed_linking()
    import tempfile

    import numpy as np
    from ase.units import fs, kJ, mol, nm
    from plumed import Plumed

    with tempfile.NamedTemporaryFile(delete=True, mode="w+") as tmp:
        plumed = Plumed()
        ps = 1000 * fs
        plumed.cmd("setMDEnergyUnits", mol / kJ)
        plumed.cmd("setMDLengthUnits", 1 / nm)
        plumed.cmd("setMDTimeUnits", 1 / ps)
        plumed.cmd("setMDChargeUnits", 1.0)
        plumed.cmd("setMDMassUnits", 1.0)

        plumed.cmd("setNatoms", len(atoms))
        # plumed.cmd("setMDEngine", "ASE")
        plumed.cmd("setLogFile", tmp.name)
        # plumed.cmd("setTimestep", float(timestep))
        plumed.cmd("setRestart", True)
        # plumed.cmd("setKbT", float(kT))
        plumed.cmd("init")
        plumed_input = remove_comments_printflush(plumed_input)
        for line in plumed_input.split("\n"):
            plumed.cmd("readInputLine", line)

        if atoms.pbc.any():
            plumed.cmd("setBox", np.asarray(atoms.cell))

        # set positions
        energy = np.zeros((1,))
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        plumed.cmd("setStep", 0)
        plumed.cmd("setPositions", atoms.get_positions())
        plumed.cmd("setMasses", atoms.get_masses())
        plumed.cmd("setForces", forces)
        plumed.cmd("setVirial", virial)
        plumed.cmd("prepareCalc")
        plumed.cmd("performCalcNoUpdate")
        plumed.cmd("getBias", energy)
    stress = virial / atoms.get_volume()
    return energy, forces, stress


class PlumedCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        plumed_input: str,
        *input_files: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plumed_input = plumed_input
        for input_file in input_files:
            assert Path(input_file).exists()
        self.input_files = input_files

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        energy, forces, stress = evaluate_plumed(
            atoms, self.plumed_input, *self.input_files
        )
        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "free_energy": energy,
        }


class PlumedHamiltonian(Hamiltonian):
    def __init__(
        self, plumed_input: str, *input_files: Union[str, Path, File, DataFuture]
    ):
        super().__init__()
        self.plumed_input = plumed_input

        # double check that supplied files are present in bare input
        bare_input = remove_comments_printflush(plumed_input)
        for i, input_file in enumerate(input_files):
            if type(input_file) is File:
                assert input_file.filepath in bare_input
            elif type(input_file) is DataFuture:
                assert input_file.file_obj.filepath in bare_input
            else:
                input_files[i] = File(input_file)
        self.input_files = input_files

        self.evaluate_app = python_app(evaluate_function, executors=["default_htex"])

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "plumed_input": self.plumed_input,
        }

    @staticmethod
    def load_calculators(
        data: list[FlowAtoms],
        *input_files: Union[File],
        plumed_input: str = "",
    ) -> tuple[list[PlumedCalculator], np.ndarray]:
        import numpy as np

        from psiflow.hamiltonians._plumed import PlumedCalculator

        calculator = PlumedCalculator(
            plumed_input,
            *[input_file.filepath for input_file in input_files],
        )
        return [calculator], np.zeros(len(data), dtype=int)
