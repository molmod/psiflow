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
    report = '\n'
    report += '\te3nn:\t'
    try:
        import e3nn
        report += e3nn.__version__
    except ModuleNotFoundError:
        report += 'module not found'

    report += '\n'
    report += '\tnequip:\t'
    try:
        import nequip
        report += nequip.__version__
    except ModuleNotFoundError:
        report += 'module not found'

    report += '\n'
    report += '\tmace:\t'
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
    return report


def check_torch():
    report = '\n'
    report += '\ttorch:\t'
    try:
        import torch
        report += torch.__version__
        report += ' (cuda available: {})'.format(torch.cuda.is_available())
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def check_sampling():
    report += '\n'
    report += '\tase:\t'
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
    report += '\tyaff:\t'
    try:
        import plumed
        report += plumed.__version__
        from psiflow.sampling.bias import try_manual_plumed_linking
        report += ' (libplumedKernel.so at {})'.format(try_manual_plumed_linking())
    except ModuleNotFoundError:
        report += 'module not found'
    report += '\n'
    return report


def main():
    path_config = Path(sys.argv[1])
    assert path_config.is_file()
    path_tmp = tempfile.mkdtemp()
    shutil.rmtree(path_tmp)
    Path(path_tmp).mkdir()
    context = psiflow.load(
            path_config,
            path_tmp,
            )
    for definition in set(*list(context.definitions.values())):
        apps    = []
        if type(definition) == ModelEvaluationExecution:
            app = python_app(check_torch, executors=[definition.executor])
            apps.append(app)
            app = python_app(check_models, executors=[definition.executor])
            apps.append(app)
            app = python_app(check_sampling, executors=[definition.executor])
            apps.append(app)
            reports = [app().result() for app in apps]
            print(type(definition).__name__)
            for report in reports:
                print(report)
        elif type(definition) == ReferenceEvaluationExecution:
            pass
