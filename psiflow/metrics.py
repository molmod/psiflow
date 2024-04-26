from __future__ import annotations  # necessary for type-guarding class methods

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import typeguard
import wandb
from parsl.app.app import python_app
from parsl.data_provider.files import File

import psiflow
from psiflow.geometry import Geometry
from psiflow.models import Model

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
def _create_table(data: np.recarray, outputs: list = []) -> str:
    from prettytable import PrettyTable

    pt = PrettyTable()
    for name in data.dtype.names:
        column = getattr(data, name)
        pt.add_column(name, column)
    pt.align = "r"
    pt.float_format = "0.2"  # gets converted to %0.2f
    s = pt.get_formatted_string("text", sortby="walker_index")
    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write(s + "\n")
    return s


create_table = python_app(_create_table, executors=["default_threads"])


@typeguard.typechecked
def _parse_walker_logs(
    statuses: list[int],
    temperatures: list[float],
    times: list[float],
    errors: list[tuple[float, float]],
    states: list[Geometry],
    resets: list[bool],
    inputs: list = [],
) -> np.recarray:
    from psiflow.data import _extract_quantities

    nwalkers = len(statuses)
    assert nwalkers == len(temperatures)
    assert nwalkers == len(times)
    assert nwalkers == len(errors)
    assert nwalkers == len(states)
    assert nwalkers == len(resets)

    dtypes = [
        ("walker_index", np.int_),
        ("status", np.int_),
        ("temperature", np.single),
        ("time", np.single),
        ("e_rmse", np.single),
        ("f_rmse", np.single),
        ("reset", np.bool_),
    ]

    # check for additional columns : phase, logprob, delta, order parameters
    identifiers, phases, logprobs, deltas = _extract_quantities(
        ("identifier", "phase", "logprob", "delta"),
        atom_indices=None,
        elements=None,
        data=states,
    )
    dtypes.append(("identifier", np.int_))
    if not all([len(p) == 0 for p in phases]):
        ncharacters = max([len(p) for p in phases])
        dtypes.append(("phase", np.unicode_, ncharacters))
    if not np.all(np.isnan(logprobs)):
        dtypes.append(("logprob", np.float_, (logprobs.shape[1],)))  # max 64 characters
    if not np.all(np.isnan(deltas)):
        dtypes.append(("delta", np.float_))

    order_names = list(set([k for g in states for k in g.order]))
    for name in order_names:
        dtypes.append((name, np.float_))
    order_parameters = _extract_quantities(
        tuple(order_names),
        atom_indices=None,
        elements=None,
        data=states,
    )

    names = [dtype[0] for dtype in dtypes]

    data = np.recarray(nwalkers, dtype=np.dtype(dtypes))
    for i in range(nwalkers):
        data.walker_index[i] = i
        data.status[i] = statuses[i]
        data.temperature[i] = temperatures[i]
        data.time[i] = times[i]
        data.e_rmse[i] = errors[i][0] * 1000 / len(states[i])  # meV / atom
        data.f_rmse[i] = errors[i][1] * 1000  # meV / angstrom
        data.reset = resets[i]
        if "identifier" in names:
            data.identifier[i] = identifiers[i]
        if "phase" in names:
            data.phase[i] = phases[i]
        if "logprob" in names:
            data.logprob[i] = logprobs[i]
        if "delta" in names:
            data.delta[i] = deltas[i]
        for j, name in enumerate(order_names):
            getattr(data, name)[i] = order_parameters[j][i]
    return data


parse_walker_logs = python_app(_parse_walker_logs, executors=["default_threads"])


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
    for atoms0, atoms1 in zip(dataset0, dataset1):
        stdout = atoms0.info.get("reference_stdout", False)
        if stdout:
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
        for symbol in symbols:
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


log_dataset = python_app(_log_dataset, executors=["default_threads"])


def fix_plotly_layout(figure):
    figure.update_layout(plot_bgcolor="white", showlegend=False)
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
    text = """
<b>circle marker</b>: walker OK <br>
<b>diamond marker</b>:  walker reset <br>
symbols are colored based on average temperature
"""
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.02,  # x position
        y=0.98,  # y position
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=2,
        borderpad=7,
        bgcolor="white",
        opacity=0.7,
    )


@typeguard.typechecked
def _to_wandb(
    wandb_id: str,
    wandb_project: str,
    walker_logs: Optional[dict],
    dataset_log: Optional[dict],
    identifier_traces: Optional[dict],
):
    import os
    import tempfile

    import colorcet as cc
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd
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

    # convert dataset_log to pandas dataframe and add identifier tracing
    customdata = []
    identifiers = dataset_log["identifier"]
    for index in identifiers:
        customdata.append(identifier_traces.get(index, (np.nan,) * 5))
    customdata = np.array(customdata, dtype=np.float32)
    dataset_log["iteration"] = customdata[:, 0]
    dataset_log["walker_index"] = customdata[:, 1]
    dataset_log["nsteps"] = customdata[:, 2]
    dataset_log["marker_symbol"] = 1 * customdata[:, 3]
    dataset_log["temperature"] = customdata[:, 4]
    df = pd.DataFrame.from_dict(dataset_log)

    df_na = df[df["temperature"].isna()]
    df_not_na = df[df["temperature"].notna()]

    # sort to get markers right!
    df_not_na = df_not_na.sort_values(by="marker_symbol")
    df_na = df_na.sort_values(by="marker_symbol")

    cmap = cc.cm.CET_I1
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, cmap.N)]

    if dataset_log is not None:
        for x_axis in dataset_log:
            if x_axis.startswith("CV") or (x_axis == "identifier"):
                for y_axis in dataset_log:
                    if (y_axis == "e_rmse") or y_axis.startswith("f_rmse"):
                        figure_ = px.scatter(
                            data_frame=df_na,
                            x=x_axis,
                            y=y_axis,
                            custom_data=[
                                "iteration",
                                "walker_index",
                                "nsteps",
                                "identifier",
                                "marker_symbol",
                            ],
                            symbol="marker_symbol",
                            symbol_sequence=["circle", "star-diamond"],
                            color_discrete_sequence=["darkgray"],
                        )
                        figure = px.scatter(
                            data_frame=df_not_na,
                            x=x_axis,
                            y=y_axis,
                            custom_data=[
                                "iteration",
                                "walker_index",
                                "nsteps",
                                "identifier",
                                "marker_symbol",
                                "temperature",
                            ],
                            symbol="marker_symbol",
                            symbol_sequence=["circle", "star-diamond"],
                            color="temperature",
                            color_continuous_scale=colors,
                        )
                        for trace in figure_.data:
                            figure.add_trace(trace)
                        figure.update_traces(  # wandb cannot deal with lines in non-circle symbols!
                            marker={"size": 12},
                            selector=dict(marker_symbol="star-diamond"),
                        )
                        figure.update_traces(
                            marker={
                                "size": 10,
                                "line": dict(width=1.0, color="DarkSlateGray"),
                            },
                            selector=dict(marker_symbol="circle"),
                        )
                        figure.update_traces(
                            hovertemplate=(
                                "<b>iteration</b>: %{customdata[0]}<br>"
                                + "<b>walker index</b>: %{customdata[1]}<br>"
                                + "<b>steps</b>: %{customdata[2]}<br>"
                                + "<b>identifier</b>: %{customdata[3]}<br>"
                                + "<extra></extra>"
                            ),
                        )
                        figure.update_xaxes(type="linear")
                        title = "dataset_" + y_axis + "_" + x_axis
                        fix_plotly_layout(figure)
                        if "e_rmse" in y_axis:
                            figure.update_layout(
                                yaxis_title="<b>energy RMSE [meV/atom]</b>"
                            )
                        else:
                            figure.update_layout(
                                yaxis_title="<b>forces RMSE [meV/atom]</b>"
                            )
                        figure.update_layout(xaxis_title="<b>" + x_axis + "</b>")
                        figures[title] = figure
    os.environ["WANDB_SILENT"] = "True"
    path_wandb = Path(tempfile.mkdtemp())
    # id is only unique per project
    wandb.init(id=wandb_id, dir=path_wandb, project=wandb_project, resume="allow")
    wandb.log(figures)
    wandb.finish()


to_wandb = python_app(_to_wandb, executors=["default_threads"])


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
        self.identifier_traces = {}
        self.iteration = 0

    def as_dict(self):
        return {
            "wandb_group": self.wandb_group,
            "wandb_project": self.wandb_project,
            "wandb_id": self.wandb_id,
        }

    def insert_name(self, model: Model):
        model.config_raw["wandb_project"] = self.wandb_project
        model.config_raw["wandb_group"] = self.wandb_group
