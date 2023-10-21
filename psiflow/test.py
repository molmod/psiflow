import sys
import shutil
from pathlib import Path

from parsl.app.app import python_app

import psiflow
from psiflow.execution import ReferenceEvaluation


def check_models():
    report = '\te3nn:\t\t'
    try:
        import e3nn
        report += e3nn.__version__
    except ModuleNotFoundError:
        report += 'module not found'

    report += '\n'
    report += '\tnequip:\t\t'
    try:
        import nequip
        report += nequip.__version__
    except ModuleNotFoundError:
        report += 'module not found'

    report += '\n'
    report += '\tmace:\t\t'
    try:
        import mace
        report += mace.__version__
    except ModuleNotFoundError:
        report += 'module not found'

    report += '\n'
    report += '\tallegro:\t'
    try:
        import allegro
        report += allegro.__version__
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'

    report += '\twandb:\t\t'
    try:
        import os
        import wandb
        report += wandb.__version__
        if 'WANDB_API_KEY' in os.environ.keys():
            report += ' (using WANDB_API_KEY={})'.format(os.environ['WANDB_API_KEY'])
        else:
            report += ' (env variable WANDB_API_KEY not found)'
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def check_torch():
    report = '\ttorch:\t'
    try:
        import torch
        report += torch.__version__
        report += ' (cuda available: {})'.format(torch.cuda.is_available())
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def check_walkers():
    report = '\tase:\t'
    try:
        import ase
        report += ase.__version__
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    report += '\tmolmod:\t'
    try:
        import molmod
        report += molmod.__version__
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    report += '\tyaff:\t'
    try:
        import yaff
        report += yaff.__version__
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    report += '\tplumed:\t'
    try:
        import plumed
        from psiflow.walkers.bias import try_manual_plumed_linking
        report += 'libplumedKernel.so at {}'.format(try_manual_plumed_linking())
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def check_reference(mpi_command):
    import subprocess
    report = '\tpymatgen:\t\t'
    try:
        import pymatgen
        report += 'OK'
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    report += '\tCP2K executable:\t'
    report += subprocess.run(['which', 'cp2k.psmp'], capture_output=True, text=True).stdout
    report += '\n'
    report += '\tMPI  executable:\t'
    report += subprocess.run(['which', mpi_command], capture_output=True, text=True).stdout
    report += '\n'
    return report


def main():
    path_config = Path(sys.argv[1])
    assert path_config.is_file()
    path_tmp = Path.cwd().resolve() / '.psiflow_internal'
    if path_tmp.is_dir():
        shutil.rmtree(path_tmp)
    Path(path_tmp).mkdir()
    context = psiflow.load(path_config, path_tmp)
    executors = {}
    for definition in context.definitions:
        print('EXECUTOR "{}":'.format(definition.name()))
        apps    = []
        if not type(definition) == ReferenceEvaluation:
            app = python_app(check_torch, executors=[definition.name()])
            apps.append(app)
            app = python_app(check_models, executors=[definition.name()])
            apps.append(app)
            app = python_app(check_walkers, executors=[definition.name()])
            apps.append(app)
            reports = [app().result() for app in apps]
            for report in reports:
                print(report)
        else:
            app = python_app(check_reference, executors=[definition.name()])
            mpi_command  = definition.mpi_command(1234).split(' ')[0]
            print(app(mpi_command).result())

