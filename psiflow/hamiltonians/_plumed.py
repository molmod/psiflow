from __future__ import annotations  # necessary for type-guarding class methods

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

import psiflow
from psiflow.geometry import Geometry
from psiflow.hamiltonians.hamiltonian import Hamiltonian, evaluate_function
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
def set_path_in_plumed(plumed_input: str, keyword: str, path_to_set: str) -> str:
    lines = plumed_input.split("\n")
    for i, line in enumerate(lines):
        if keyword in line:
            if "FILE=" not in line:
                lines[i] = line + " FILE={}".format(path_to_set)
                continue
            line_before = line.split("FILE=")[0]
            line_after = line.split("FILE=")[1].split()[1:]
            lines[i] = (
                line_before + "FILE={} ".format(path_to_set) + " ".join(line_after)
            )
    return "\n".join(lines)


@typeguard.typechecked
@psiflow.serializable
class PlumedHamiltonian(Hamiltonian):
    _plumed_input: str
    external: Optional[psiflow._DataFuture]

    def __init__(
        self,
        plumed_input: str,
        external: Union[None, str, Path, File, DataFuture] = None,
    ):
        super().__init__()
        _plumed_input = remove_comments_printflush(plumed_input)

        if type(external) in [str, Path]:
            external = File(str(external))
        if external is not None:
            assert external.filepath in _plumed_input
            _plumed_input = _plumed_input.replace(external.filepath, "PLACEHOLDER")
        self._plumed_input = _plumed_input
        self.external = external
        self._create_apps()

    def _create_apps(self):
        self.evaluate_app = python_app(evaluate_function, executors=["default_htex"])

    def __eq__(self, other: Hamiltonian) -> bool:
        if type(other) is not type(self):
            return False
        if self.plumed_input() != other.plumed_input():
            return False
        return True

    def plumed_input(self):
        plumed_input = self._plumed_input
        if self.external is not None:
            plumed_input = plumed_input.replace("PLACEHOLDER", self.external.filepath)
        return plumed_input

    def serialize_calculator(self):
        if self.external is not None:
            external = self.external.filepath
        else:
            external = self.external
        return dump_json(
            hamiltonian=self.__class__.__name__,
            plumed_input=self.plumed_input(),
            external=external,
            inputs=[self.external],  # wait for them to complete
            outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
        ).outputs[0]

    @staticmethod
    def deserialize_calculator(plumed_input: str, external: Optional[str]) -> Any:
        from psiflow.hamiltonians.utils import PlumedCalculator

        return PlumedCalculator(plumed_input, external)

    @property
    def parameters(self: Hamiltonian) -> dict:
        return {
            "plumed_input": self.plumed_input(),
        }

    @staticmethod
    def load_calculators(
        data: list[Geometry],
        external: Optional[File],
        plumed_input: str = "",
    ) -> tuple[list[Any], np.ndarray]:
        import numpy as np

        from psiflow.hamiltonians.utils import PlumedCalculator

        calculator = PlumedCalculator(plumed_input, external)
        return [calculator], np.zeros(len(data), dtype=int)
