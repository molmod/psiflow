import sys
import shutil
import tempfile
import numpy as np
from pathlib import Path

from parsl.app.app import python_app

import psiflow
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution


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

    report += '\twandb:\t'
    try:
        import wandb
        report += wandb.__version__
        wandb.cli.cli.login()
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


def check_sampling():
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
        #report += plumed.__version__
        from psiflow.sampling.bias import try_manual_plumed_linking
        report += 'libplumedKernel.so at {}'.format(try_manual_plumed_linking())
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def check_reference(mpi_command, cp2k_command):
    import subprocess
    report = '\tpymatgen:\t\t'
    try:
        import pymatgen
        report += 'OK'
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    report += '\tCP2K executable:\t'
    report += subprocess.run(['which', cp2k_command], capture_output=True, text=True).stdout
    report += '\n'
    report += '\tMPI  executable:\t'
    report += subprocess.run(['which', mpi_command], capture_output=True, text=True).stdout
    report += '\n'
    return report


def main():
    path_config = Path(sys.argv[1])
    assert path_config.is_file()
    path_tmp = Path.cwd().resolve() / '.psiflow_internal'
    shutil.rmtree(path_tmp)
    Path(path_tmp).mkdir()
    context = psiflow.load(
            path_config,
            path_tmp,
            )
    executors = {}
    for cls_, definitions in context.definitions.items():
        for definition in definitions:
            executor = definition.executor
            if executor not in executors.keys():
                executors[executor] = []
            executors[executor].append(definition)
    for executor, definitions in executors.items():
        print('EXECUTOR "{}":'.format(executor))
        for definition in set(definitions):
            apps    = []
            if not type(definition) == ReferenceEvaluationExecution:
                app = python_app(check_torch, executors=[definition.executor])
                apps.append(app)
                app = python_app(check_models, executors=[definition.executor])
                apps.append(app)
                app = python_app(check_sampling, executors=[definition.executor])
                apps.append(app)
                reports = [app().result() for app in apps]
                for report in reports:
                    print(report)
            else:
                app = python_app(check_reference, executors=[definition.executor])
                mpi_command  = definition.mpi_command(1234).split(' ')[0]
                cp2k_command = definition.cp2k_exec
                print(app(mpi_command, cp2k_command).result())

