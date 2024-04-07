from __future__ import annotations  # necessary for type-guarding class methods

import typeguard
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.hamiltonians._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.utils import copy_app_future


def _shift_time(inputs: list = [], outputs: list = []) -> None:
    from pathlib import Path

    if Path(inputs[0].filepath).exists():
        with open(inputs[0], "r") as f:
            content = f.read()
    else:
        content = ""
    lines = content.split("\n")
    delta_time = 1e-10
    for i in range(len(lines)):
        if i < 3:
            continue  # header
        time = i * delta_time
        lines[i] = "\t".join([str(time), *lines[i].split()[1:]])
    with open(outputs[0], "w") as f:
        f.write("\n".join(lines))


shift_time = python_app(_shift_time, executors=["default_threads"])


@typeguard.typechecked
class Metadynamics:
    def __init__(self, plumed_input: str):
        assert "METAD" in plumed_input
        if "RESTART" not in plumed_input:
            plumed_input = "RESTART\n" + plumed_input
        plumed_input = remove_comments_printflush(plumed_input)
        self.hillsfile = psiflow.context().new_file("hills_", ".txt")
        plumed_input = set_path_in_plumed(
            plumed_input,
            "METAD",
            self.hillsfile.filepath,
        )
        self.plumed_input = plumed_input

    def input(self) -> AppFuture:
        # hillsfile = shift_time(
        #        inputs=[self.hillsfile],
        #        outputs=[psiflow.context().new_file('hills_', '.txt')],
        #        ).outputs[0]
        self.plumed_input = set_path_in_plumed(
            self.plumed_input,
            "METAD",
            self.hillsfile.filepath,
        )
        # self.hillsfile = hillsfile
        return copy_app_future(self.plumed_input, inputs=[self.hillsfile])

    def wait_for(self, result: AppFuture) -> None:
        self.hillsfile = copy_app_future(
            0,
            inputs=[result, self.hillsfile],
            outputs=[File(self.hillsfile.filepath)],
        ).outputs[0]

    def reset(self) -> None:
        self.hillsfile = psiflow.context().new_file("hills_", ".txt")
        self.plumed_input = set_path_in_plumed(
            self.plumed_input,
            "METAD",
            self.hillsfile.filepath,
        )

    def __eq__(self, other) -> bool:
        if type(other) is not Metadynamics:
            return False
        return self.plumed_input == other.plumed_input

    def copy(self) -> Metadynamics:
        return Metadynamics(str(self.plumed_input))
