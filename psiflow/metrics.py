from __future__ import annotations  # necessary for type-guarding class methods

import logging
import os
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
import typeguard
import wandb
from parsl.app.app import python_app
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset, FlowAtoms, NullState
from psiflow.models import BaseModel

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _save_walker_logs(data: dict[str, list], path: Path) -> str:
    from prettytable import PrettyTable

    pt = PrettyTable()
    field_names = [
        "walker_index",
        "counter",
        "is_reset",
        "e_rsme",
        "f_rmse",
        "disagreement",
        "temperature",
        "identifier",
    ]
    for key in data:
        if (key not in field_names) and not (key == "stdout"):
            field_names.append(key)
    field_names.append("stdout")
    for key in field_names:
        if key not in data:
            continue
        pt.add_column(key, data[key])
    pt.align = "r"
    pt.float_format = "0.2"  # gets converted to %0.2f
    s = pt.get_formatted_string("text", sortby="walker_index")
    with open(path, "w") as f:
        f.write(s + "\n")
    return s


save_walker_logs = python_app(_save_walker_logs, executors=["Default"])


@typeguard.typechecked
def _save_dataset_log(data: dict[str, list], path: Path) -> str:
    from prettytable import PrettyTable

    pt = PrettyTable()
    field_names = [
        "identifier",
        "e_rmse",
        "f_rmse",
    ]
    for key in data:
        if (key not in field_names) and not (key == "stdout"):
            field_names.append(key)
    field_names.append("stdout")
    for key in field_names:
        if key not in data:
            continue
        pt.add_column(key, data[key])
    pt.align = "r"
    pt.float_format = "0.2"  # gets converted to %0.2f
    s = pt.get_formatted_string("text", sortby="identifier")
    with open(path, "w") as f:
        f.write(s + "\n")
    return s


save_dataset_log = python_app(_save_dataset_log, executors=["Default"])


@typeguard.typechecked
def _log_walker(
    walker_index: int,
    evaluated_state: FlowAtoms,
    error: tuple[Optional[float], Optional[float]],
    condition: bool,
    identifier: int,
    disagreement: Optional[float] = None,
    **metadata,
) -> NamedTuple:
    from pathlib import Path

    data = {}
    data["walker_index"] = walker_index
    data["counter"] = metadata["counter"]
    data["is_reset"] = condition
    data["e_rmse"] = error[0]
    data["f_rmse"] = error[1]
    data["disagreement"] = disagreement
    data["temperature"] = metadata.get("temperature", None)

    if not evaluated_state == NullState:
        data["identifier"] = identifier - 1
    else:
        data["identifier"] = None

    for name in metadata["state"].info:
        if name.startswith("CV"):
            data[name] = metadata["state"].info[name]

    if "stdout" in metadata:
        data["stdout"] = Path(metadata["stdout"]).stem
    else:
        data["stdout"] = None
    return data


log_walker = python_app(_log_walker, executors=["Default"])


@typeguard.typechecked
def _gather_walker_logs(*walker_data: dict) -> dict[str, list]:
    data = {}
    columns = list(set([v for wd in walker_data for v in wd.keys()]))
    for key in columns:
        values = []
        for wd in walker_data:
            values.append(wd.get(key, None))
        data[key] = values
    return data


gather_walker_logs = python_app(_gather_walker_logs, executors=["Default"])


@typeguard.typechecked
def _log_dataset(inputs: list[File] = []) -> dict[str, list]:
    import numpy as np
    from ase.data import chemical_symbols

    from psiflow.data import read_dataset
    from psiflow.utils import compute_error, get_index_element_mask

    dataset0 = read_dataset(slice(None), inputs=[inputs[0]])
    dataset1 = read_dataset(slice(None), inputs=[inputs[1]])
    assert len(dataset0) == len(dataset1)

    data = {}

    # define x axis
    x_axis = ["identifier"]
    for atoms in dataset0:
        assert "identifier" in atoms.info
        for key in atoms.info:
            if key.startswith("CV"):
                x_axis.append(key)
    x_axis = list(set(x_axis))
    for key in x_axis:
        if key not in data:
            data[key] = []
        for atoms in dataset0:
            data[key].append(atoms.info.get(key, None))

    # define y axis
    _all = [set(a.numbers) for a in dataset0]
    numbers = sorted(list(set(b for a in _all for b in a)))
    symbols = [chemical_symbols[n] for n in numbers]
    y_axis = ["e_rmse", "f_rmse"]
    y_axis += ["f_rmse_{}".format(s) for s in symbols]

    # define data array and fill
    for key in y_axis:
        data[key] = []
    data["stdout"] = []
    for i, (atoms0, atoms1) in enumerate(zip(dataset0, dataset1)):
        stdout = atoms0.info.get("reference_stdout", None)
        if stdout is not None:
            stdout = Path(stdout).stem
        else:
            stdout = "NA"
        data["stdout"].append(stdout)
        data["e_rmse"].append(
            compute_error(
                atoms0,
                atoms1,
                "rmse",
                mask=np.array([True] * len(atoms0)),
                properties=["energy"],
            )[0]
        )
        data["f_rmse"].append(
            compute_error(
                atoms0,
                atoms1,
                "rmse",
                mask=np.array([True] * len(atoms0)),
                properties=["forces"],
            )[0]
        )
        for j, symbol in enumerate(symbols):
            mask = get_index_element_mask(
                atoms0.numbers,
                elements=[symbol],
                atom_indices=None,
            )
            data["f_rmse_{}".format(symbol)].append(
                compute_error(
                    atoms0,
                    atoms1,
                    "rmse",
                    mask=mask,
                    properties=["forces"],
                )[0]
            )
    assert len(list(data.keys())) == len(x_axis) + len(y_axis) + 1
    return data


log_dataset = python_app(_log_dataset, executors=["Default"])


def fix_plotly_layout(figure):
    figure.update_layout(plot_bgcolor="white")
    figure.update_xaxes(
        mirror=True,
        ticks="inside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    figure.update_yaxes(
        mirror=True,
        ticks="inside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickformat=".1f",
    )


@typeguard.typechecked
def _to_wandb(
    wandb_id: str,
    wandb_project: str,
    walker_logs: Optional[dict],
    dataset_log: Optional[dict],
):
    import os
    import tempfile

    import plotly.express as px
    import wandb

    figures = {}
    if walker_logs is not None:
        y = np.array(walker_logs["f_rmse"])
        for key in walker_logs:
            if key == "walker_index":
                name = "index"
                tickformat = ".0f"
            elif key.startswith("CV"):
                name = key
                tickformat = ".2f"
            else:
                name = None
                tickformat = None
            if name is not None:
                x = np.array(walker_logs[key])
                figure = px.scatter(x=x, y=y)
                figure.update_xaxes(type="linear", tickformat=tickformat)
                title = "walkers_f_rmse_" + name
                fix_plotly_layout(figure)
                figure.update_layout(yaxis_title="forces RMSE [meV/A]")
                figures[title] = figure
    if dataset_log is not None:
        for x_axis in dataset_log:
            if x_axis.startswith("CV") or (x_axis == "identifier"):
                for y_axis in dataset_log:
                    if (y_axis == "e_rmse") or y_axis.startswith("f_rmse"):
                        x = dataset_log[x_axis]
                        y = dataset_log[y_axis]
                        figure = px.scatter(x=x, y=y)
                        figure.update_xaxes(type="linear")
                        title = "dataset_" + y_axis + "_" + x_axis
                        fix_plotly_layout(figure)
                        if "e_rmse" in y_axis:
                            figure.update_layout(yaxis_title="energy RMSE [meV/atom]")
                        else:
                            figure.update_layout(yaxis_title="forces RMSE [meV/atom]")
                        figures[title] = figure
    os.environ["WANDB_SILENT"] = "True"
    path_wandb = Path(tempfile.mkdtemp())
    # id is only unique per project
    wandb.init(id=wandb_id, dir=path_wandb, project=wandb_project, resume="allow")
    wandb.log(figures)
    wandb.finish()


to_wandb = python_app(_to_wandb, executors=["Default"])


@typeguard.typechecked
class Metrics:
    def __init__(
        self,
        wandb_group: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_id: Optional[str] = None,
    ) -> None:
        self.wandb_group = wandb_group
        self.wandb_project = wandb_project
        self.wandb_name = "main"
        self.wandb_id = None
        if self.wandb_group is not None:
            os.environ["WANDB_SILENT"] = "True"
            assert "WANDB_API_KEY" in os.environ
            if self.wandb_id is None:
                self.wandb_id = wandb.sdk.lib.runid.generate_id()
                resume = None
            else:
                resume = "must"
            wandb.init(
                id=self.wandb_id,
                project=self.wandb_project,
                group=self.wandb_group,
                name=self.wandb_name,
                dir=psiflow.context().path,
                resume=resume,
            )
        self.walker_logs = []

    def as_dict(self):
        return {
            "wandb_group": self.wandb_group,
            "wandb_project": self.wandb_project,
            "wandb_id": self.wandb_id,
        }

    def insert_name(self, model: BaseModel):
        model.config_raw["wandb_project"] = self.wandb_project
        model.config_raw["wandb_group"] = self.wandb_group

    def log_walker(
        self,
        i,
        walker,
        metadata,
        state,
        error,
        condition,
        identifier,
        disagreement=None,
    ):
        # log walker total counter value instead
        # of counter from metadata
        metadata_dict = metadata._asdict()
        metadata_dict["counter"] = walker.counter
        log = log_walker(
            i,
            state,
            error,
            condition,
            identifier,
            disagreement,
            **metadata_dict,
        )
        self.walker_logs.append(log)

    def save(
        self,
        path: Union[str, Path],
        model: Optional[BaseModel] = None,
        dataset: Optional[Dataset] = None,
    ):
        path = Path(path)
        if not path.exists():
            path.mkdir()
        walker_logs = None
        dataset_log = None
        if len(self.walker_logs) > 0:
            walker_logs = gather_walker_logs(*self.walker_logs)
            save_walker_logs(walker_logs, path / "walkers.log")
            self.walker_logs = []
        if model is not None:
            assert dataset is not None
            inputs = [dataset.data_future, model.evaluate(dataset).data_future]
            dataset_log = log_dataset(inputs=inputs)
            save_dataset_log(dataset_log, path / "dataset.log")
        if self.wandb_group is not None:
            f = to_wandb(self.wandb_id, self.wandb_project, walker_logs, dataset_log)
