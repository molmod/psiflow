from __future__ import annotations  # necessary for type-guarding class methods

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from ase.calculators.calculator import Calculator, all_changes
from ase.units import fs, kJ, mol, nm
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.data import FlowAtoms
from psiflow.hamiltonians.hamiltonian import Hamiltonian
from psiflow.hamiltonians.utils import check_forces, evaluate_function
from psiflow.utils import dump_json

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
class PlumedCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        plumed_input: str,
        *input_files: Union[str, Path],
        max_force: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plumed_input = plumed_input
        for input_file in input_files:
            assert Path(input_file).exists()
        self.input_files = input_files
        self.max_force = max_force

        self.tmp = tempfile.NamedTemporaryFile(prefix="plumed_", mode="w+")
        # plumed creates a back up when this file would already exist
        os.remove(self.tmp.name)

        from plumed import Plumed

        plumed = Plumed()
        ps = 1000 * fs
        plumed.cmd("setMDEnergyUnits", mol / kJ)
        plumed.cmd("setMDLengthUnits", 1 / nm)
        plumed.cmd("setMDTimeUnits", 1 / ps)
        plumed.cmd("setMDChargeUnits", 1.0)
        plumed.cmd("setMDMassUnits", 1.0)

        # plumed.cmd("setMDEngine", "ASE")
        plumed.cmd("setLogFile", self.tmp.name)
        # plumed.cmd("setTimestep", float(timestep))
        plumed.cmd("setRestart", True)
        self.plumed = plumed
        self.initialize_plumed = True  # need natoms to initialize

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        if self.initialize_plumed:
            self.plumed.cmd("setNatoms", len(atoms))
            self.plumed.cmd("init")
            plumed_input = remove_comments_printflush(self.plumed_input)
            for line in plumed_input.split("\n"):
                self.plumed.cmd("readInputLine", line)
            self.initialize_plumed = False
            os.remove(self.tmp.name)  # remove whatever plumed has created

        if atoms.pbc.any():
            self.plumed.cmd("setBox", np.asarray(atoms.cell))

        # set positions
        energy = np.zeros((1,))
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        self.plumed.cmd("setStep", 0)
        self.plumed.cmd("setPositions", atoms.get_positions())
        self.plumed.cmd("setMasses", atoms.get_masses())
        self.plumed.cmd("setForces", forces)
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        self.plumed.cmd("getBias", energy)
        stress = virial / atoms.get_volume()

        if self.max_force is not None:
            check_forces(forces, atoms, self.max_force)
        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "free_energy": energy,
        }


@typeguard.typechecked
class PlumedHamiltonian(Hamiltonian):
    def __init__(
        self, plumed_input: str, *input_files: Union[str, Path, File, DataFuture]
    ):
        super().__init__()
        self.plumed_input = plumed_input

        # double check that supplied files are present in bare input
        bare_input = remove_comments_printflush(plumed_input)
        self.input_files = []
        for input_file in input_files:
            if type(input_file) is File:
                assert input_file.filepath in bare_input
                assert Path(input_file.filepath).exists()
            elif type(input_file) is DataFuture:
                assert input_file.file_obj.filepath in bare_input
            else:
                assert type(input_file) in [str, Path]
                if type(input_file) is str:
                    input_file = Path.cwd() / input_file
                assert input_file.exists()
                input_file = File(input_file)
            self.input_files.append(input_file)

        self.evaluate_app = python_app(evaluate_function, executors=["default_htex"])

    def __eq__(self, other: Hamiltonian) -> bool:
        if type(other) is not type(self):
            return False
        if self.plumed_input != other.plumed_input:
            return False
        if len(self.input_files) != len(other.input_files):
            return False
        for file, file_ in zip(self.input_files, other.input_files):
            if file.filepath != file_.filepath:
                return False
        return True

    def serialize(self):
        input_files = [file.filepath for file in self.input_files]
        return dump_json(
            hamiltonian=self.__class__.__name__,
            plumed_input=self.plumed_input,
            input_files=input_files,
            inputs=self.input_files,  # wait for them to complete
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize(plumed_input: str, input_files: list[str]) -> PlumedCalculator:
        return PlumedCalculator(plumed_input, *input_files)

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
