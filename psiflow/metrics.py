from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, NamedTuple, Type, Any
from collections import namedtuple
import os
import logging
import pandas
import typeguard
import wandb
from pathlib import Path
import numpy as np

from parsl.app.app import python_app
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset, FlowAtoms
from psiflow.models import BaseModel
from psiflow.walkers import BaseWalker, DynamicWalker


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
def _save_walkers_report(data: dict[str, list], path: Path) -> str:
    from prettytable import PrettyTable
    pt = PrettyTable()
    field_names = [
            'walker_index',
            'counter',
            'is_reset',
            'e_rsme',
            'f_rmse',
            'disagreement',
            'temperature',
            'identifier',
            ]
    for key in data:
        if (key not in field_names) and not (key == 'stdout'):
            field_names.append(key)
    field_names.append('stdout')
    for key in field_names:
        if key not in data:
            continue
        pt.add_column(key, data[key])
    pt.align = 'r'
    pt.float_format = '0.2' # gets converted to %0.2f
    s = pt.get_formatted_string('text', sortby='walker_index')
    with open(path, 'w') as f:
        f.write(s)
    return s
save_walkers_report = python_app(_save_walkers_report, executors=['Default'])


@typeguard.typechecked
def _save_dataset_report(data: dict[str, list], path: Path) -> str:
    from prettytable import PrettyTable
    pt = PrettyTable()
    field_names = [
            'identifier',
            'e_mae',
            'e_rmse',
            'f_rmse',
            ]
    for key in data:
        if (key not in field_names) and not (key == 'stdout'):
            field_names.append(key)
    field_names.append('stdout')
    for key in field_names:
        if key not in data:
            continue
        pt.add_column(key, data[key])
    pt.align = 'r'
    pt.float_format = '0.2' # gets converted to %0.2f
    s = pt.get_formatted_string('text', sortby='identifier')
    with open(path, 'w') as f:
        f.write(s)
    return s
save_dataset_report = python_app(_save_dataset_report, executors=['Default'])


@typeguard.typechecked
def _report_walker(
        walker_index: int,
        evaluated_state: Optional[FlowAtoms],
        error: tuple[Optional[float], Optional[float]],
        condition: bool,
        identifier: int,
        disagreement: Optional[float] = None,
        **metadata,
        ) -> NamedTuple:
    from pathlib import Path
    data = {}
    data['walker_index'] = walker_index
    data['is_reset'] = condition
    data['counter'] = metadata['counter']
    data['temperature'] = metadata.get('temperature', None)
    data['e_rmse'] = error[0]
    data['f_rmse'] = error[1]
    data['disagreement'] = disagreement

    if evaluated_state is not None:
        data['identifier'] = identifier
    else:
        data['identifier'] = None

    for name in metadata['state'].info:
        if name.startswith('CV'):
            data[name] = metadata['state'].info[name]

    if 'stdout' in metadata['state'].info:
        data['stdout'] = Path(metadata['state'].info['stdout']).stem
    else:
        data['stdout'] = None

    return data
report_walker = python_app(_report_walker, executors=['Default'])


@typeguard.typechecked
def _gather_walker_reports(*walker_data: dict) -> dict[str, list]:
    import numpy as np
    data = {}
    columns = list(set([v for wd in walker_data for v in wd.keys()]))
    for key in columns:
        values = []
        for wd in walker_data:
            values.append(wd.get(key, None))
        data[key] = values
    return data
gather_walker_reports = python_app(_gather_walker_reports, executors=['Default'])


@typeguard.typechecked
def _report_dataset(inputs: list[File] = []) -> dict[str, list]:
    import pandas
    import numpy as np
    from ase.data import chemical_symbols
    from psiflow.data import read_dataset
    from psiflow.utils import compute_error, get_index_element_mask
    dataset0 = read_dataset(slice(None), inputs=[inputs[0]])
    dataset1 = read_dataset(slice(None), inputs=[inputs[1]])
    assert len(dataset0) == len(dataset1)

    data = {}

    # define x axis
    x_axis = ['identifier']
    for atoms in dataset0:
        assert 'identifier' in atoms.info
        for key in atoms.info:
            if key.startswith('CV'):
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
    y_axis  = ['e_mae', 'e_rmse', 'f_rmse']
    y_axis += ['f_rmse_{}'.format(s) for s in symbols]

    # define data array and fill
    #data = np.zeros((len(dataset0), len(x_axis) + len(y_axis)))
    for key in y_axis:
        data[key] = []
    for i, (atoms0, atoms1) in enumerate(zip(dataset0, dataset1)):
        data['e_mae'].append(compute_error(
                atoms0,
                atoms1,
                'mae',
                mask=np.array([True] * len(atoms0)),
                properties=['energy'],
                )[0])
        data['e_rmse'].append(compute_error(
                atoms0,
                atoms1,
                'rmse',
                mask=np.array([True] * len(atoms0)),
                properties=['energy'],
                )[0])
        data['f_rmse'].append(compute_error(
                atoms0,
                atoms1,
                'rmse',
                mask=np.array([True] * len(atoms0)),
                properties=['forces'],
                )[0])
        for j, symbol in enumerate(symbols):
            mask = get_index_element_mask(
                    atoms0.numbers,
                    elements=[symbol],
                    atom_indices=None,
                    )
            data['f_rmse_{}'.format(symbol)].append(compute_error(
                atoms0,
                atoms1,
                'rmse',
                mask=mask,
                properties=['forces'],
                )[0])
    assert len(list(data.keys())) == len(x_axis) + len(y_axis)
    return data
report_dataset = python_app(_report_dataset, executors=['Default'])


def fix_plotly_layout(figure):
    figure.update_layout(plot_bgcolor='white')
    figure.update_xaxes(
            mirror=True,
            ticks='inside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            )
    figure.update_yaxes(
            mirror=True,
            ticks='inside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            tickformat='.1f',
            )


@typeguard.typechecked
def _log_wandb(
        wandb_id: str,
        walkers_report: Optional[dict],
        dataset_report: Optional[dict],
        ):
    import wandb
    import plotly.express as px
    figures = {}
    if walkers_report is not None:
        y = np.array(walkers_report['f_rmse'])
        for key in walkers_report:
            if key == 'walker_index':
                name = 'index'
                tickformat = '.0f'
            elif key.startswith('CV'):
                name = key
                tickformat = '.2f'
            else:
                name = None
                tickformat = None
            if name is not None:
                x = np.array(walkers_report[key])
                figure = px.scatter(x=x, y=y)
                figure.update_xaxes(type='linear', tickformat=tickformat)
                title = 'walkers_f_rmse_' + name
                fix_plotly_layout(figure)
                figure.update_layout(yaxis_title='forces RMSE [meV/A]')
                figures[title] = figure
    if dataset_report is not None:
        for x_axis in dataset_report:
            if x_axis.startswith('CV') or (x_axis == 'identifier'):
                for y_axis in dataset_report:
                    if (y_axis in ['e_rmse', 'e_mae']) or y_axis.startswith('f_rmse'):
                        x = dataset_report[x_axis]
                        y = dataset_report[y_axis]
                        figure = px.scatter(x=x, y=y)
                        figure.update_xaxes(type='linear')
                        title = 'dataset_' + y_axis + '_' + x_axis
                        fix_plotly_layout(figure)
                        if 'e_mae' in y_axis:
                            figure.update_layout(yaxis_title='energy MAE [meV/atom]')
                        elif 'e_rmse' in y_axis:
                            figure.update_layout(yaxis_title='energy RMSE [meV/atom]')
                        else:
                            figure.update_layout(yaxis_title='forces RMSE [meV/atom]')
                        figures[title] = figure
    wandb.init(id=wandb_id, resume='must')
    wandb.log(figures)
    wandb.finish()
log_wandb = python_app(_log_wandb, executors=['Default'])


@typeguard.typechecked
class Metrics:

    def __init__(
            self,
            wandb_project: Optional[str] = None,
            wandb_group: Optional[str] = None,
            wandb_id: Optional[str] = None,
            ) -> None:
        self.wandb_project = wandb_project
        self.wandb_group   = wandb_group
        self.wandb_id      = None
        if wandb_project is not None:
            assert 'WANDB_API_KEY' in os.environ
            if self.wandb_id is None:
                self.wandb_id = wandb.sdk.lib.runid.generate_id()
                resume = None
            else:
                resume = 'must'
            wandb.init(
                    id=self.wandb_id,
                    project=self.wandb_project,
                    group=self.wandb_group,
                    dir=psiflow.context().path,
                    resume=resume,
                    )

        self.walker_reports = []

    def log_walker(self, i, metadata, state, error, condition, identifier):
        report = report_walker(
                i,
                state,
                error,
                condition,
                identifier,
                **metadata._asdict(),
                )
        self.walker_reports.append(report)

    def save(
            self,
            path: Union[str, Path],
            model: Optional[BaseModel] = None,
            dataset: Optional[Dataset] = None,
            ):
        path = Path(path)
        if not path.exists():
            path.mkdir()
        walkers_report = None
        dataset_report = None
        if len(self.walker_reports) > 0:
            walkers_report = gather_walker_reports(*self.walker_reports)
            save_walkers_report(walkers_report, path / 'walkers.log')
        if model is not None:
            assert dataset is not None
            inputs = [dataset.data_future, model.evaluate(dataset).data_future]
            dataset_report = report_dataset(inputs=inputs)
            save_dataset_report(dataset_report, path / 'dataset.log')
        if self.wandb_id is not None:
            f = log_wandb(self.wandb_id, walkers_report, dataset_report)
