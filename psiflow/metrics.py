from __future__ import annotations  # necessary for type-guarding class methods

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import typeguard
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.geometry import Geometry
from psiflow.hamiltonians import Hamiltonian
from psiflow.models import Model
from psiflow.sampling import SimulationOutput
from psiflow.utils.apps import combine_futures, log_message, setup_logger

logger = setup_logger(__name__)


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
def _parse_walker_log(
    statuses: list[int],
    temperatures: list[Optional[float]],
    times: list[Optional[float]],
    errors: list[np.ndarray],
    states: list[Geometry],
    resets: list[bool],
    inputs: list = [],
) -> np.recarray:
    from psiflow.data.utils import _extract_quantities

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
        ("identifier", np.int_),  # useful, but also needed for proper dataset logging
        ("reset", np.bool_),
    ]

    # check for additional columns : phase, logprob, delta, order parameters
    identifiers, phases, logprobs, deltas = _extract_quantities(
        ("identifier", "phase", "logprob", "delta"),
        None,
        None,
        *states,
    )
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
        None,
        None,
        *states,
    )

    names = [dtype[0] for dtype in dtypes]

    data = np.recarray(nwalkers, dtype=np.dtype(dtypes))
    for i in range(nwalkers):
        data.walker_index[i] = i
        data.status[i] = statuses[i]
        data.temperature[i] = temperatures[i]
        data.time[i] = times[i]
        data.e_rmse[i] = errors[i][0]
        data.f_rmse[i] = errors[i][1]
        data.reset[i] = resets[i]
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


parse_walker_log = python_app(_parse_walker_log, executors=["default_threads"])


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
individual circles represent a single snapshot in the training/validation set <br>
circles are colored based on the error difference between the current and previous model
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
    wandb_api_key: str,
    inputs: list = [],
) -> None:
    import os

    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_SILENT"] = "True"
    import tempfile
    import numpy as np
    from pathlib import Path

    import colorcet as cc
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import wandb

    from psiflow.metrics import fix_plotly_layout
    from psiflow.utils.io import _load_metrics

    Y_AXES = [
        "delta",
        "phase",
        "logprob",
        "e_rmse",
        "f_rmse",
        "iteration",
        "temperature",
        "time",
        "status",
        "reset",
    ]

    metrics = _load_metrics(inputs=[inputs[0]])

    data_names = []
    for name in metrics.dtype.names:
        m = getattr(metrics, name)
        if m.dtype == np.float_ and not np.all(np.isnan(m)):
            pass
        elif m.dtype.kind == "U" and any([len(s) for s in m]):
            pass
        elif m.dtype == bool and np.any(m):
            pass
        elif m.dtype == np.int_:
            pass
        else:
            continue
        data_names.append(name)

    # create data arrays with and without delta_e / delta_f color coding
    metrics.e_rmse *= 1000  # meV/atom
    metrics.f_rmse *= 1000  # meV/angstrom
    # mask_e = np.logical_not(np.isnan(metrics.e_rmse[:, 0]))
    # mask_f = np.logical_not(np.isnan(metrics.f_rmse[:, 0]))
    # total_e_rmse = np.sqrt(np.mean(metrics.e_rmse[mask_e] ** 2))
    # total_f_rmse = np.sqrt(np.mean(metrics.f_rmse[mask_f] ** 2))

    data = {name: getattr(metrics, name) for name in data_names}
    data["delta_e_rmse"] = metrics.e_rmse[:, 0] - metrics.e_rmse[:, 1]
    data["delta_f_rmse"] = metrics.f_rmse[:, 0] - metrics.f_rmse[:, 1]
    data["e_rmse"] = metrics.e_rmse[:, 0]  # ensure arrays are one dimensional
    data["f_rmse"] = metrics.f_rmse[:, 0]
    df = pd.DataFrame.from_dict(data)

    # convert missing strings to empty string, not 0.0
    for name in metrics.dtype.names:
        if name in df.columns and getattr(metrics, name).dtype.kind == "U":
            df[name] = df[name].replace("", np.nan).fillna("")

    cmap = cc.cm.CET_D4
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, cmap.N)]

    figures = {}
    for name in data_names:
        if name not in Y_AXES:  # it's an x axis
            x_axis = name
            for y in ["e_rmse", "f_rmse"]:
                key = "delta_" + y
                # df_na = df[df[key].isna()]
                df_not_na = df[df[key].notna()]

                hover_names = []
                hover_template = ""
                for name in data_names:
                    if name in [x_axis, y]:
                        continue
                    if getattr(metrics, name).dtype == np.float_:
                        format_str = ":.2f"
                    elif getattr(metrics, name).dtype == np.int_:
                        format_str = ":d"
                    elif getattr(metrics, name).dtype.kind == "U":
                        format_str = ""
                    else:
                        format_str = ""
                    index = len(hover_names)
                    s = f"<b>{name}</b>: %{{customdata[{index}]{format_str}}}<br>"
                    hover_template += s
                    hover_names.append(name)

                figure = px.scatter(
                    data_frame=df_not_na,
                    x=x_axis,
                    y=y,
                    custom_data=hover_names,
                    color=key,
                    color_continuous_scale=colors,
                    symbol_sequence=["circle"],
                )
                figure.update_traces(
                    marker={
                        "size": 10,
                        "line": dict(width=1.0, color="DarkSlateGray"),
                        "symbol": "circle",
                    },
                )

                figure.update_traces(hovertemplate=hover_template)
                figure.update_xaxes(type="linear")
                figure.update_yaxes(type="linear")
                title = y + "_" + x_axis
                fix_plotly_layout(figure)
                if y == "e_rmse":
                    figure.update_layout(yaxis_title="<b>energy RMSE [meV/atom]</b>")
                else:
                    figure.update_layout(yaxis_title="<b>forces RMSE [meV/atom]</b>")
                figure.update_layout(xaxis_title="<b>" + x_axis + "</b>")
                figures[title] = figure
    path_wandb = Path(tempfile.mkdtemp())
    wandb.init(id=wandb_id, dir=path_wandb, project=wandb_project, resume="allow")
    wandb.log(figures)
    wandb.finish()
    return None


to_wandb = python_app(_to_wandb, executors=["default_htex"])


def reconstruct_dtypes(dtype):
    dtypes = []
    for name in dtype.names:
        field_dtype = dtype[name]
        # Check if the field has a shape (for multidimensional data)
        if field_dtype.shape:
            dtypes.append((name, field_dtype.base, field_dtype.shape))
        else:
            # Handling unicode/string by specifying the fixed size explicitly
            if np.issubdtype(field_dtype, np.string_) or np.issubdtype(
                field_dtype, np.unicode_
            ):
                # Get character length for unicode/string types
                char_length = (
                    field_dtype.itemsize
                    if np.issubdtype(field_dtype, np.string_)
                    else field_dtype.itemsize // 4
                )
                dtypes.append((name, field_dtype.type, char_length))
            else:
                dtypes.append((name, field_dtype.type))
    return dtypes


@typeguard.typechecked
def _add_walker_log(
    walker_log: np.recarray,
    inputs: list = [],
    outputs: list = [],
) -> None:
    from numpy.lib.recfunctions import stack_arrays

    from psiflow.utils.io import _load_metrics, _save_metrics

    # only add 'successful' states in metrics
    # everything else was already logged (and printed to stdout) using parse_walker_log
    walker_log = walker_log[walker_log.identifier != -1]
    walker_log = np.sort(walker_log, order="identifier")

    # currently assumes that walkers don't fundamentally change!
    # i.e. that reconstruct_dtypes(walker_log.dtype) yields the same
    # set of dtypes
    if os.path.isfile(inputs[0]) and os.path.getsize(inputs[0]) > 0:
        metrics = _load_metrics(inputs=[inputs[0]])
        dtype = metrics.dtype
        metrics = stack_arrays(  # np.concatenate is not compatible with recarray
            (metrics, np.recarray(len(walker_log), dtype=dtype)),
            asrecarray=True,
            usemask=False,
        )
        start = len(metrics) - len(walker_log)
    else:  # initialize dtype based on walkers
        dtypes = reconstruct_dtypes(walker_log.dtype)
        for i, dtype in enumerate(list(dtypes)):
            if dtype[0] in ["e_rmse", "f_rmse"]:
                dtypes[i] = (dtype[0], np.float_, (2,))
        dtype = np.dtype(dtypes)
        metrics = np.recarray(len(walker_log), dtype=dtype)
        start = 0

    for i in range(len(walker_log)):
        metrics_i = start + i
        for name in metrics.dtype.names:
            if name in ["e_rmse", "f_rmse"]:
                getattr(metrics, name)[metrics_i, 0] = getattr(walker_log, name)[i]
                getattr(metrics, name)[metrics_i, 1] = np.nan
            elif name == "walker_index":
                getattr(metrics, name)[metrics_i] = walker_log.walker_index[i]
            else:
                getattr(metrics, name)[metrics_i] = getattr(walker_log, name)[i]

    _save_metrics(metrics, outputs=[outputs[0]])


add_walker_log = python_app(_add_walker_log, executors=["default_threads"])


@typeguard.typechecked
def _update_logs(
    inputs: list = [],
    outputs: list = [],
):
    from psiflow.data.utils import _compute_rmse, _extract_quantities, _read_frames
    from psiflow.utils.io import _load_metrics, _save_metrics

    data0 = _read_frames(inputs=[inputs[1]])
    data1 = _read_frames(inputs=[inputs[2]])

    energy0 = _extract_quantities(("per_atom_energy",), None, None, *data0)[0]
    energy1 = _extract_quantities(("per_atom_energy",), None, None, *data1)[0]
    e_rmse = _compute_rmse(energy0, energy1, reduce=False)

    forces0 = _extract_quantities(("forces",), None, None, *data0)[0]
    forces1 = _extract_quantities(("forces",), None, None, *data1)[0]
    f_rmse = _compute_rmse(forces0, forces1, reduce=False)

    identifiers = _extract_quantities(("identifier",), None, None, *data0)[0]
    assert np.all(identifiers >= 0)
    indices = np.argsort(identifiers)

    e_rmse = e_rmse[indices]
    f_rmse = f_rmse[indices]

    metrics = _load_metrics(inputs=[inputs[0]])
    assert np.allclose(metrics.identifier, identifiers[indices])  # check order
    metrics.e_rmse[:, 1] = metrics.e_rmse[:, 0]
    metrics.e_rmse[:, 0] = e_rmse
    metrics.f_rmse[:, 1] = metrics.f_rmse[:, 0]
    metrics.f_rmse[:, 0] = f_rmse
    _save_metrics(metrics, outputs=[outputs[0]])


update_logs = python_app(_update_logs, executors=["default_threads"])


@typeguard.typechecked
def _initialize_wandb(
    wandb_group: str,
    wandb_project: str,
    wandb_name: str,
    wandb_id: Optional[str],
    wandb_api_key: str,
    path_wandb: Path,
) -> str:
    import os

    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_SILENT"] = "True"
    import wandb

    if wandb_id is None:
        wandb_id = wandb.sdk.lib.runid.generate_id()
        resume = None
    else:
        resume = "must"
    wandb.init(
        id=wandb_id,
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
        dir=path_wandb,
        resume=resume,
    )
    return wandb_id


initialize_wandb = python_app(_initialize_wandb, executors=["default_htex"])


@typeguard.typechecked
@psiflow.serializable
class Metrics:
    wandb_group: Optional[str]
    wandb_project: Optional[str]
    wandb_id: Optional[str]
    metrics: psiflow._DataFuture

    def __init__(
        self,
        wandb_group: Optional[str] = None,
        wandb_project: Optional[str] = None,
        metrics: Optional[psiflow._DataFuture] = None,
    ) -> None:
        self.wandb_group = wandb_group
        self.wandb_project = wandb_project
        if wandb_group is not None and wandb_project is not None:
            assert "WANDB_API_KEY" in os.environ
            self.wandb_id = initialize_wandb(
                wandb_group,
                wandb_project,
                "metrics",
                None,
                os.environ["WANDB_API_KEY"],
                psiflow.context().path,
            ).result()
        else:
            self.wandb_id = None

        if metrics is None:
            metrics = psiflow.context().new_file("metrics_", ".numpy")

        self.metrics = metrics

    def insert_name(self, model: Model):
        model.config_raw["wandb_project"] = self.wandb_project
        model.config_raw["wandb_group"] = self.wandb_group

    def log_walkers(
        self,
        outputs: list[SimulationOutput],
        errors: list[Union[AppFuture, np.ndarray]],
        states: list[Union[AppFuture, Geometry]],
        resets: list[Union[AppFuture, bool]],
    ):
        statuses = [o.status for o in outputs]
        temperatures = [o.temperature for o in outputs]
        times = [o.time for o in outputs]
        walker_log = parse_walker_log(
            combine_futures(inputs=statuses),
            combine_futures(inputs=temperatures),
            combine_futures(inputs=times),
            combine_futures(inputs=errors),
            combine_futures(inputs=states),
            combine_futures(inputs=resets),
        )
        log_message(logger, "\n{}", create_table(walker_log))
        metrics = add_walker_log(
            walker_log,
            inputs=[self.metrics],
            outputs=[psiflow.context().new_file("metrics_", ".numpy")],
        ).outputs[0]
        self.metrics = metrics

    def update(self, data: Dataset, hamiltonian: Hamiltonian):
        data_ = data.evaluate(hamiltonian)
        metrics = update_logs(
            inputs=[self.metrics, data.extxyz, data_.extxyz],
            outputs=[psiflow.context().new_file("metrics_", ".numpy")],
        ).outputs[0]
        self.metrics = metrics
        if self.wandb_id is not None:
            self.to_wandb()

    def to_wandb(self):
        if self.wandb_group is None and self.wandb_project is None:
            raise ValueError("initialize Metrics with wandb group and project names")
        assert self.wandb_id is not None
        return to_wandb(
            self.wandb_id,
            self.wandb_project,
            os.environ['WANDB_API_KEY'],
            inputs=[self.metrics],
        )
