import os
import tempfile
import yaff
import molmod
import numpy as np

from parsl.app.app import python_app, join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

from flower.execution import Container, ModelExecutionDefinition
from flower.utils import _new_file, copy_data_future


def try_manual_plumed_linking():
    if 'PLUMED_KERNEL' not in os.environ.keys():
        # try linking manually
        if 'CONDA_PREFIX' in os.environ.keys(): # for conda environments
            p = 'CONDA_PREFIX'
        elif 'PREFIX' in os.environ.keys(): # for pip environments
            p = 'PREFIX'
        else:
            print('failed to set plumed .so kernel')
            pass
        path = os.environ[p] + '/lib/libplumedKernel.so'
        if os.path.exists(path):
            os.environ['PLUMED_KERNEL'] = path
            print('plumed kernel manually set at at : {}'.format(path))


def set_path_in_plumed(plumed_input, keyword, path_to_set):
    lines = plumed_input.split('\n')
    for i, line in enumerate(lines):
        if keyword in line.split():
            line_before = line.split('FILE=')[0]
            line_after  = line.split('FILE=')[1].split()[1:]
            lines[i] = line_before + 'FILE={} '.format(path_to_set) + ' '.join(line_after)
    return '\n'.join(lines)


def parse_plumed_input(plumed_input):
    allowed_keywords = ['METAD', 'RESTRAINT', 'EXTERNAL', 'UPPER_WALLS']
    found = False
    for key in allowed_keywords:
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if key in line.split():
                assert not found
                cv = line.split('ARG=')[1].split()[0]
                #label = line.split('LABEL=')[1].split()[0]
                bias = (key, cv)
                found = True
    return bias


def generate_external_grid(bias_function, cv, cv_label, periodic=False):
    _periodic = 'false' if not periodic else 'true'
    grid = ''
    grid += '#! FIELDS {} external.bias der_{}\n'.format(cv_label, cv_label)
    grid += '#! SET min_{} {}\n'.format(cv_label, np.min(cv))
    grid += '#! SET max_{} {}\n'.format(cv_label, np.max(cv))
    grid += '#! SET nbins_{} {}\n'.format(cv_label, len(cv))
    grid += '#! SET periodic_{} {}\n'.format(cv_label, _periodic)
    for i in range(len(cv)):
        grid += '{} {} {}\n'.format(cv[i], bias_function(cv[i]), 0)
    return grid


def evaluate_bias(plumed_input, keyword, cv, inputs=[]):
    import tempfile
    import os
    import numpy as np
    import yaff
    yaff.log.set_level(yaff.log.silent)
    import molmod
    from flower.sampling.utils import ForcePartASE, create_forcefield, \
            ForceThresholdExceededException
    from flower.sampling.bias import try_manual_plumed_linking
    from flower.data import read_dataset
    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    values = np.zeros((len(dataset), 2)) # column 0 for CV, 1 for bias
    system = yaff.System(
            numbers=dataset[0].get_atomic_numbers(),
            pos=dataset[0].get_positions() * molmod.units.angstrom,
            rvecs=dataset[0].get_cell() * molmod.units.angstrom,
            )
    try_manual_plumed_linking()
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    colvar_log = tmp.name # dummy log file
    plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
    plumed_input += '\nPRINT STRIDE=1 ARG={} FILE={}'.format(cv, colvar_log)
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write(plumed_input) # write input
        path_input = f.name
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    plumedlog = tmp.name # dummy log file
    part_plumed = yaff.external.ForcePartPlumed(
            system,
            timestep=1000*molmod.units.femtosecond, # does this matter?
            restart=0,
            fn=path_input,
            fn_log=plumedlog,
            )
    ff = yaff.pes.ForceField(system, [part_plumed])
    for i, atoms in enumerate(dataset):
        ff.update_pos(atoms.get_positions() * molmod.units.angstrom)
        ff.update_rvecs(atoms.get_cell() * molmod.units.angstrom)
        part_plumed.plumedstep = i
        values[i, 1] = ff.compute() / molmod.units.kjmol
        part_plumed.plumed.cmd('update')
    part_plumed.plumed.cmd('update') # flush last
    values[:, 0] = np.loadtxt(colvar_log)[:, 1]
    os.unlink(plumedlog)
    os.unlink(colvar_log)
    os.unlink(path_input)
    return values


class PlumedBias(Container):
    """Represents a PLUMED bias potential"""

    def __init__(self, context, plumed_input, data_future=None):
        super().__init__(context)
        assert 'PRINT' not in plumed_input
        self.plumed_input = plumed_input
        self.keyword, self.cv = parse_plumed_input(plumed_input)
        if data_future is None:
            self.data_future = File(_new_file(context.path, 'bias_', '.txt'))
        else:
            assert (isinstance(data_future, DataFuture) or isinstance(data_future, File))
            self.data_future = data_future

    def evaluate(self, dataset):
        plumed_input = self.prepare_input()
        return self.context.apps(PlumedBias, 'evaluate')(
                plumed_input,
                self.keyword,
                self.cv,
                inputs=[dataset.data_future, self.data_future],
                )

    def prepare_input(self):
        return self.plumed_input

    def copy(self):
        return PlumedBias(
                self.context,
                self.plumed_input,
                data_future=copy_data_future(
                    inputs=[self.data_future],
                    outputs=[File(_new_file(self.context.path, 'bias_', '.txt'))],
                    ).outputs[0]
                )

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ModelExecutionDefinition].executor_label
        app_evaluate = python_app(evaluate_bias, executors=[executor_label])
        context.register_app(cls, 'evaluate', app_evaluate)


class MetadynamicsBias(PlumedBias):

    def __init__(self, context, plumed_input, data_future=None):
        super().__init__(context, plumed_input, data_future)
        assert self.keyword == 'METAD'

    def prepare_input(self):
        plumed_input = str(self.plumed_input)
        plumed_input = set_path_in_plumed(plumed_input, 'METAD', self.data_future.filepath)
        plumed_input = 'RESTART\n' + plumed_input
        plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
        return plumed_input


class ExternalBias(PlumedBias):

    def __init__(self, context, plumed_input, data_future=None):
        super().__init__(context, plumed_input, data_future)
        assert self.keyword == 'EXTERNAL'

    def prepare_input(self):
        plumed_input = str(self.plumed_input)
        plumed_input = set_path_in_plumed(plumed_input, 'EXTERNAL', self.data_future.filepath)
        print(plumed_input)
        with open(self.data_future.filepath, 'r') as f:
            print(f.read())
        return plumed_input


def create_bias(context, plumed_input, path_data=None, data=None):
    keyword, cv = parse_plumed_input(plumed_input)
    if isinstance(path_data, str):
        assert data is None
        assert os.path.exists(path_data)
        data_future = File(path_data) # convert to File before passing it as future
    elif isinstance(data, str):
        assert path_data is None
        path_data = _new_file(context.path, 'bias_', '.txt')
        with open(path_data, 'w') as f:
            f.write(data)
        data_future = File(path_data)
    else:
        data_future = None
    if (keyword == 'RESTRAINT' or keyword == 'UPPER_WALLS'):
        return PlumedBias(context, plumed_input)
    elif keyword == 'METAD':
        return MetadynamicsBias(context, plumed_input, data_future=data_future)
    elif keyword == 'EXTERNAL':
        return ExternalBias(context, plumed_input, data_future=data_future)
    else:
        raise ValueError('plumed keyword {} unrecognized'.format(keyword))
