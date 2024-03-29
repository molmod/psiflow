from __future__ import annotations  # necessary for type-guarding class methods

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import typeguard
from ase import Atoms
from parsl.app.app import join_app, python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset, FlowAtoms, read_dataset, write_dataset
from psiflow.utils import copy_data_future, save_txt

logger = logging.getLogger(__name__)  # logging per module


PLUMED_BIAS_KEYWORDS = [
    "METAD",
    "RESTRAINT",
    "MOVINGRESTRAINT",
    "EXTERNAL",
    "UPPER_WALLS",
    "LOWER_WALLS",
]


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
            print("plumed kernel manually set at : {}".format(path))
        else:
            raise ValueError("plumed kernel not found at {}".format(path))
    return os.environ["PLUMED_KERNEL"]


@typeguard.typechecked
def set_path_in_plumed(plumed_input: str, keyword: str, path_to_set: str) -> str:
    lines = plumed_input.split("\n")
    for i, line in enumerate(lines):
        if keyword in line.split():
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
def parse_plumed_input(plumed_input: str) -> tuple[list[tuple], tuple[str, ...]]:
    components = []
    variables = set()
    metad_exists = False
    external_exists = False
    for key in PLUMED_BIAS_KEYWORDS:
        lines = plumed_input.split("\n")
        for line in lines:
            if key in line.split():
                if key == "METAD":
                    assert not metad_exists, "only one METAD action is allowed"
                    metad_exists = True
                if key == "EXTERNAL":
                    assert not external_exists, "only one EXTERNAL action is allowed"
                    external_exists = True
                args = line.split("ARG=")[1].split()[0]
                args = tuple(args.split(","))
                for variable in args:
                    variables.add(variable)
                components.append((key, args))
    nvariables = len(variables)
    assert nvariables > 0, "define at least one CV"
    return components, tuple(sorted(variables))


@typeguard.typechecked
def generate_external_grid(
    bias_function: Callable,
    variable: np.ndarray,
    variable_label: str,
    periodic: bool = False,
) -> str:
    _periodic = "false" if not periodic else "true"
    grid = ""
    grid += "#! FIELDS {} external.bias der_{}\n".format(variable_label, variable_label)
    grid += "#! SET min_{} {}\n".format(variable_label, np.min(variable))
    grid += "#! SET max_{} {}\n".format(variable_label, np.max(variable))
    grid += "#! SET nbins_{} {}\n".format(variable_label, len(variable))
    grid += "#! SET periodic_{} {}\n".format(variable_label, _periodic)
    for i in range(len(variable)):
        grid += "{} {} {}\n".format(variable[i], bias_function(variable[i]), 0)
    return grid


@typeguard.typechecked
def evaluate_bias(
    plumed_input: str,
    variables: tuple[str, ...],
    inputs: list[File] = [],
) -> np.ndarray:
    import os
    import tempfile

    import numpy as np
    import yaff

    yaff.log.set_level(yaff.log.silent)
    import molmod

    from psiflow.data import read_dataset
    from psiflow.walkers.bias import try_manual_plumed_linking
    from psiflow.walkers.molecular_dynamics_yaff import ForcePartPlumed

    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    values = np.zeros((len(dataset), len(variables) + 1))  # column 0 for CV, 1 for bias
    if dataset[0].pbc.all():
        rvecs = dataset[0].get_cell() * molmod.units.angstrom
    else:
        rvecs = None
    system = yaff.System(
        numbers=dataset[0].get_atomic_numbers(),
        pos=dataset[0].get_positions() * molmod.units.angstrom,
        rvecs=rvecs,
    )
    try_manual_plumed_linking()
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
    tmp.close()
    colvar_log = tmp.name  # dummy log file
    plumed_input += "\nFLUSH STRIDE=1"  # has to come before PRINT?!
    plumed_input += "\nPRINT STRIDE=1 ARG={} FILE={}".format(
        ",".join(variables), colvar_log
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
    tmp.close()
    plumed_log = tmp.name  # dummy log file
    with tempfile.NamedTemporaryFile(delete=False, mode="w+") as f:
        f.write(plumed_input)  # write input
        path_input = f.name
    part_plumed = ForcePartPlumed(
        system,
        timestep=1 * molmod.units.femtosecond,  # does not matter
        restart=1,
        fn=path_input,
        fn_log=plumed_log,
    )
    ff = yaff.pes.ForceField(system, [part_plumed])
    for i, atoms in enumerate(dataset):
        ff.update_pos(atoms.get_positions() * molmod.units.angstrom)
        if rvecs is not None:
            assert atoms.pbc.all()
            ff.update_rvecs(atoms.get_cell() * molmod.units.angstrom)
        else:
            assert not atoms.pbc.all()
        values[i, -1] = ff.compute() / molmod.units.kjmol
        part_plumed.plumed.cmd("update")
        part_plumed.plumedstep = 3  # can be anything except zero; pick a prime
    if len(dataset) > 1:  # counter weird behavior
        part_plumed.plumed.cmd("update")  # flush last
    values[:, :-1] = np.loadtxt(colvar_log).reshape(-1, len(variables) + 1)[:, 1:]
    if "MOVINGRESTRAINT" in plumed_input:
        values[:, -1] = 0.0  # bias has no meaning
    os.unlink(plumed_log)
    os.unlink(colvar_log)
    os.unlink(path_input)
    return values


app_evaluate = python_app(evaluate_bias, executors=["default_htex"])


@typeguard.typechecked
def _gather_partitions(
    shape: tuple[int, int],
    partitions: dict[str, list[int]],
    **values: np.ndarray,
) -> np.ndarray:
    final = np.zeros(shape)
    final[:] = np.nan
    for key, partition in partitions.items():
        for i, index in enumerate(partition):
            final[index, :] = values[key][i, :]
    return final


gather_partitions = python_app(_gather_partitions, executors=["default_threads"])


@join_app
@typeguard.typechecked
def partitioned_evaluate_bias(
    plumed_input: str,
    variables: tuple[str, ...],
    data_future: File,
    inputs: list[File] = [],
) -> AppFuture:
    import psiflow
    from psiflow.data import NullState, read_dataset, write_dataset
    from psiflow.walkers.bias import app_evaluate

    dataset = read_dataset(slice(None), inputs=[data_future])

    def hash_atoms(atoms):
        return tuple(atoms.numbers) + tuple(atoms.pbc)

    partitions = {}
    for i, atoms in enumerate(dataset):
        if atoms == NullState:
            continue
        key = hash_atoms(atoms)
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(i)

    futures = {}
    for key, partition in partitions.items():
        data_future = psiflow.context().new_file("data_", ".xyz")
        write_dataset(
            [dataset[i] for i in partition],
            outputs=[data_future],
        )
        futures[key] = app_evaluate(
            plumed_input,
            variables,
            inputs=[data_future] + inputs[1:],
        )

    # use strings as dict keys in order to pass them as kwargs
    keys = {key: str(i) for i, key in enumerate(partitions.keys())}
    partitions_ = {keys[key]: partitions[key] for key in keys}
    futures_ = {keys[key]: futures[key] for key in keys}
    return gather_partitions(
        (len(dataset), len(variables) + 1),
        partitions_,
        **futures_,
    )


@typeguard.typechecked
def _reset_mtd(
    condition: bool,
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> None:
    if condition:  # empty hills file
        content = " "
    else:  # copy existing hills
        with open(inputs[0], "r") as f:
            content = f.read()
    with open(outputs[0], "w") as f:
        f.write(content)


reset_mtd = python_app(_reset_mtd, executors=["default_threads"])


@typeguard.typechecked
def extract_grid(
    index: int,
    values: np.ndarray,
    targets: np.ndarray,
    slack: Optional[float] = None,
) -> list[int]:
    import numpy as np

    nstates = len(targets)
    variable_values = values[:, index]
    if slack is None:
        slack = 1e10  # infinite slack
    deltas = np.abs(targets[:, np.newaxis] - variable_values[np.newaxis, :])
    indices = np.argmin(deltas, axis=1)
    found = np.abs(targets - variable_values[indices]) < slack
    to_extract = []  # create list of indices to extract
    for i in range(nstates):
        if found[i]:
            index = indices[i]
            to_extract.append(int(index))
        else:
            raise ValueError(
                "could not find state for target value {}".format(targets[i])
            )
    assert len(to_extract) == nstates
    return to_extract


app_extract_grid = python_app(extract_grid, executors=["default_threads"])


@typeguard.typechecked
def extract_between(
    index: int,
    values: np.ndarray,
    min_value: float,
    max_value: float,
) -> list[int]:
    import numpy as np

    variable_values = values[:, index]
    mask_higher = variable_values > min_value
    mask_lower = variable_values < max_value
    indices = np.arange(len(variable_values))[mask_higher * mask_lower]
    return [int(index) for index in indices]


app_extract_between = python_app(extract_between, executors=["default_threads"])


@typeguard.typechecked
def extract_column(array: np.ndarray, index: int) -> np.ndarray:
    assert index < array.shape[1]
    return array[:, index].reshape(-1, 1)  # maintain shape


app_extract_column = python_app(extract_column, executors=["default_threads"])


@typeguard.typechecked
def insert_cv_values(
    variables: tuple[str, ...],
    state: Union[FlowAtoms, Atoms],
    values: np.ndarray,
) -> Union[FlowAtoms, Atoms]:
    from psiflow.data import NullState

    if state == NullState:
        return state
    assert len(values.shape) == 2
    assert values.shape[0] == 1
    assert values.shape[1] == len(variables) + 1
    for i, variable in enumerate(variables):
        state.info[variable] = values[0, i]
    return state


app_insert_cv_values = python_app(insert_cv_values, executors=["default_threads"])


@typeguard.typechecked
def insert_cv_values_data(
    variables: tuple[str, ...],
    values: np.ndarray,
    inputs: list[File] = [],
    outputs: list[File] = [],
) -> None:
    data = read_dataset(slice(None), inputs=[inputs[0]])
    assert len(data) == values.shape[0]
    for i, atoms in enumerate(data):
        insert_cv_values(variables, atoms, values[i, :].reshape(1, -1))
    write_dataset(data, outputs=[outputs[0]])


app_insert_cv_values_data = python_app(
    insert_cv_values_data, executors=["default_threads"]
)


@typeguard.typechecked
class PlumedBias:
    """Represents a PLUMED bias potential"""

    keys_with_future = ["EXTERNAL", "METAD"]

    def __init__(self, plumed_input: str, data: Optional[dict] = None):
        assert "ENERGY=kj/mol" in plumed_input, (
            "please set the PLUMED energy " "units to kj/mol"
        )
        assert "..." not in plumed_input, (
            "combine each of the PLUMED actions " "into a single line in the input file"
        )
        assert "__FILL__" not in plumed_input, "__FILL__ is not supported"
        if ("PRINT" in plumed_input) or ("FLUSH" in plumed_input):
            logger.info(
                "removing *all* print and flush statements "
                "in the input to avoid generating additional (untracked) "
                "files"
            )
        if "MOVINGRESTRAINT" in plumed_input:
            assert "STEP0=0" in plumed_input
        plumed_input = remove_comments_printflush(plumed_input)
        components, variables = parse_plumed_input(plumed_input)
        assert len(variables) > 0
        assert len(components) > 0
        self.variables = variables
        self.components = components
        self.plumed_input = plumed_input

        # initialize data future for METAD and EXTERNAL components
        context = psiflow.context()
        self.data_futures = OrderedDict()
        if data is None:
            data = {}
        else:
            for key, value in data.items():
                assert key in self.keys
                if type(value) is str:
                    self.data_futures[key] = save_txt(
                        value,
                        outputs=[context.new_file(key + "_", ".txt")],
                    ).outputs[0]
                else:
                    assert isinstance(value, DataFuture) or isinstance(value, File)
                    self.data_futures[key] = value
        for key in self.keys:
            if (key not in self.data_futures.keys()) and (
                key in PlumedBias.keys_with_future
            ):
                assert key != "EXTERNAL"  # has to be initialized by user
                self.data_futures[key] = save_txt(
                    " ",
                    outputs=[context.new_file(key + "_", ".txt")],
                ).outputs[0]
        if "METAD" in self.keys:
            self.data_futures.move_to_end("METAD", last=False)  # put it first

    def evaluate(
        self, dataset: Dataset, variable: Optional[str] = None, as_dataset: bool = False
    ) -> Union[AppFuture, Dataset]:
        plumed_input = self.prepare_input()
        lines = plumed_input.split("\n")
        for i, line in enumerate(lines):
            if "METAD" in line.split():
                assert "PACE=" in line  # PACE needs to be specified
                line_before = line.split("PACE=")[0]
                line_after = line.split("PACE=")[1].split()[1:]
                pace = 2147483647  # some random high prime number
                lines[i] = line_before + "PACE={} ".format(pace) + " ".join(line_after)
        plumed_input = "\n".join(lines)
        values = partitioned_evaluate_bias(
            plumed_input,
            self.variables,
            dataset.data_future,
            inputs=[dataset.data_future] + self.futures,
        )
        if not as_dataset:
            if variable is not None:
                assert variable in self.variables
                index = self.variables.index(variable)
                return app_extract_column(values, index)
            else:
                return values
        else:
            future = app_insert_cv_values_data(
                self.variables,
                values,
                inputs=[dataset.data_future],
                outputs=[psiflow.context().new_file("data_", ".xyz")],
            )
            return Dataset(None, future.outputs[0])

    def prepare_input(self) -> str:
        plumed_input = str(self.plumed_input)
        for key in self.keys:
            if key in ["METAD", "EXTERNAL"]:  # keys for which path needs to be set
                plumed_input = set_path_in_plumed(
                    plumed_input,
                    key,
                    self.data_futures[key].filepath,
                )
        if "METAD" in self.keys:  # necessary to print hills properly
            plumed_input = "RESTART\n" + plumed_input
            plumed_input += "\nFLUSH STRIDE=1"  # has to come before PRINT?!
        return plumed_input

    def copy(self) -> PlumedBias:
        context = psiflow.context()
        new_futures = OrderedDict()
        for key, future in self.data_futures.items():
            new_futures[key] = copy_data_future(
                inputs=[future],
                outputs=[context.new_file(key + "_", ".txt")],
            ).outputs[0]
        if len(new_futures) == 0:  # let constructor take care of it
            new_futures = None
        return PlumedBias(self.plumed_input, data=new_futures)

    def adjust_moving_restraint(
        self,
        variable: str,
        steps: Optional[int],
        kappas: Optional[tuple[float, float]],
        centers: Optional[tuple[float, float]],
    ) -> None:
        plumed_input = str(self.plumed_input)
        lines = plumed_input.split("\n")
        found = False
        for i, line in enumerate(lines):
            if "MOVINGRESTRAINT" in line.split():
                if "ARG={}".format(variable) in line.split():
                    assert not found
                    line_ = line
                    if kappas is not None:
                        for j in range(2):
                            line_before = line_.split("KAPPA{}=".format(j))[0]
                            line_after = line_.split("KAPPA{}=".format(j))[1].split()[
                                1:
                            ]
                            line_ = (
                                line_before
                                + "KAPPA{}={} ".format(j, kappas[j])
                                + " ".join(line_after)
                            )
                    if centers is not None:
                        for j in range(2):
                            line_before = line_.split("AT{}=".format(j))[0]
                            line_after = line_.split("AT{}=".format(j))[1].split()[1:]
                            line_ = (
                                line_before
                                + "AT{}={} ".format(j, centers[j])
                                + " ".join(line_after)
                            )
                    if steps is not None:
                        line_before = line_.split("STEP1=")[0]
                        line_after = line_.split("STEP1=")[1].split()[1:]
                        line_ = (
                            line_before
                            + "STEP1={} ".format(steps)
                            + " ".join(line_after)
                        )
                    lines[i] = line_
                    found = True
        assert found
        self.plumed_input = "\n".join(lines)

    def adjust_restraint(
        self, variable: str, kappa: Optional[float], center: Optional[float]
    ) -> None:
        plumed_input = str(self.plumed_input)
        lines = plumed_input.split("\n")
        found = False
        for i, line in enumerate(lines):
            if "RESTRAINT" in line.split() and "MOVING" not in line.split():
                if "ARG={}".format(variable) in line.split():
                    assert not found
                    line_ = line
                    if kappa is not None:
                        line_before = line_.split("KAPPA=")[0]
                        line_after = line_.split("KAPPA=")[1].split()[1:]
                        line_ = (
                            line_before
                            + "KAPPA={} ".format(kappa)
                            + " ".join(line_after)
                        )
                    if center is not None:
                        line_before = line_.split("AT=")[0]
                        line_after = line_.split("AT=")[1].split()[1:]
                        line_ = (
                            line_before + "AT={} ".format(center) + " ".join(line_after)
                        )
                    lines[i] = line_
                    found = True
        assert found
        self.plumed_input = "\n".join(lines)

    def get_restraint(self, variable: str) -> tuple[float, float]:
        plumed_input = str(self.plumed_input)
        lines = plumed_input.split("\n")
        found = False
        for line in lines:
            if "RESTRAINT" in line.split() and "MOVING" not in line.split():
                if "ARG={}".format(variable) in line.split():
                    assert not found
                    kappa = float(line.split("KAPPA=")[1].split()[0])
                    center = float(line.split("AT=")[1].split()[0])
                    found = True
        assert found
        return kappa, center

    def get_moving_restraint(self, variable: str) -> tuple[float, tuple, tuple]:
        plumed_input = str(self.plumed_input)
        lines = plumed_input.split("\n")
        found = False
        for line in lines:
            if "MOVINGRESTRAINT" in line.split():
                if "ARG={}".format(variable) in line.split():
                    assert not found
                    kappas = (
                        float(line.split("KAPPA0=")[1].split()[0]),
                        float(line.split("KAPPA1=")[1].split()[0]),
                    )
                    centers = (
                        float(line.split("AT0=")[1].split()[0]),
                        float(line.split("AT1=")[1].split()[0]),
                    )
                    steps = float(line.split("STEP1=")[1].split()[0])
                    found = True
        assert found
        return steps, kappas, centers

    def reset(self, condition: Union[bool, AppFuture] = None):
        if condition is None:
            condition = True
        if "METAD" in self.keys:
            self.data_futures["METAD"] = reset_mtd(
                condition,
                inputs=[self.data_futures["METAD"]],
                outputs=[psiflow.context().new_file("METAD_", ".txt")],
            ).outputs[0]

    def extract_grid(
        self,
        dataset: Dataset,
        variable: str,
        targets: np.ndarray,
        slack: Optional[float] = None,
    ) -> Dataset:
        assert variable in self.variables
        index = self.variables.index(variable)
        values = self.evaluate(dataset)
        indices = app_extract_grid(  # is future!
            index,
            values,
            targets,
            slack,
        )
        return dataset[indices]

    def extract_between(
        self,
        dataset: Dataset,
        variable: str,
        min_value: float,
        max_value: float,
    ) -> Dataset:
        assert variable in self.variables
        index = self.variables.index(variable)
        values = self.evaluate(dataset)
        indices = app_extract_between(
            index,
            values,
            min_value,
            max_value,
        )
        return dataset[indices]

    def save(
        self,
        path: Union[Path, str],
        require_done: bool = True,
    ) -> tuple[DataFuture, dict[str, DataFuture]]:
        path = Path(path)
        path.mkdir(exist_ok=True)
        path_input = path / "plumed_input.txt"
        input_future = save_txt(
            self.plumed_input,
            outputs=[File(str(path_input))],
        ).outputs[0]
        data_futures = {}
        for key in self.data_futures.keys():
            path_key = path / (key + ".txt")
            data_futures[key] = copy_data_future(
                inputs=[self.data_futures[key]],
                outputs=[File(str(path_key))],
            ).outputs[0]
        if require_done:
            input_future.result()
            for value in data_futures.values():
                value.result()
        return input_future, data_futures

    @classmethod
    def load(cls, path: Union[Path, str]) -> PlumedBias:
        path = Path(path)
        assert path.is_dir()
        path_input = path / "plumed_input.txt"
        assert path_input.is_file()
        with open(path_input, "r") as f:
            plumed_input = f.read()
        data = {}
        for key in cls.keys_with_future:
            path_key = path / (key + ".txt")
            if path_key.is_file():
                with open(path_key, "r") as f:
                    data[key] = f.read()
        return cls(plumed_input, data=data)

    @classmethod
    def from_file(cls, path_plumed: Union[Path, str]) -> PlumedBias:
        assert path_plumed.is_file()
        with open(path_plumed, "r") as f:
            plumed_input = f.read()
        return cls(plumed_input)

    @property
    def keys(self) -> list[str]:
        keys = sorted([c[0] for c in self.components])
        if "METAD" in keys:
            keys.remove("METAD")
            return ["METAD"] + keys
        else:
            return keys

    @property
    def futures(self) -> list[DataFuture]:
        return [value for _, value in self.data_futures.items()]  # MTD first

    @classmethod
    def create_apps(cls) -> None:
        pass
