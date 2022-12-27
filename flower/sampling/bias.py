import os
import tempfile
import yaff
import molmod
import numpy as np
from pathlib import Path
from collections import OrderedDict

from parsl.app.app import python_app, join_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File

from flower.execution import Container, ModelExecutionDefinition
from flower.utils import _new_file, copy_data_future, save_txt
from flower.data import read_dataset


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
    biases = []
    for key in allowed_keywords:
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if key in line.split():
                #assert not found
                cv = line.split('ARG=')[1].split()[0]
                #label = line.split('LABEL=')[1].split()[0]
                biases.append((key, cv))
    return biases


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


def evaluate_bias(plumed_input, cv, inputs=[]):
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
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    plumed_log = tmp.name # dummy log file
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write(plumed_input) # write input
        path_input = f.name
    part_plumed = yaff.external.ForcePartPlumed(
            system,
            timestep=1*molmod.units.femtosecond, # does not matter
            restart=1,
            fn=path_input,
            fn_log=plumed_log,
            )
    ff = yaff.pes.ForceField(system, [part_plumed])
    for i, atoms in enumerate(dataset):
        ff.update_pos(atoms.get_positions() * molmod.units.angstrom)
        ff.update_rvecs(atoms.get_cell() * molmod.units.angstrom)
        values[i, 1] = ff.compute() / molmod.units.kjmol
        part_plumed.plumed.cmd('update')
        part_plumed.plumedstep = 3 # can be anything except zero; pick a prime
    part_plumed.plumed.cmd('update') # flush last
    values[:, 0] = np.loadtxt(colvar_log)[:, 1]
    os.unlink(plumed_log)
    os.unlink(colvar_log)
    os.unlink(path_input)
    return values


def find_states_in_data(plumed_input, cv, cv_min, cv_max, cv_step, slack, inputs=[]):
    import numpy as np
    from flower.sampling.bias import evaluate_bias
    cv_values = evaluate_bias(plumed_input, cv, inputs=inputs)[:, 0]
    nstates = np.floor((cv_max - cv_min) / cv_step).astype(np.int32) + 1
    targets = cv_min + cv_step * np.arange(nstates)
    deltas  = np.abs(targets[:, np.newaxis] - cv_values[np.newaxis, :])
    indices = np.argmin(deltas, axis=1)
    found   = np.abs(targets - cv_values[indices]) < slack
    to_extract = [] # create list of indices to extract
    for i in range(nstates):
        if found[i]:
            index = indices[i]
            to_extract.append(int(index))
        else:
            pass
    return to_extract


class PlumedBias(Container):
    """Represents a PLUMED bias potential"""
    keys_with_future = ['EXTERNAL', 'METAD']

    def __init__(self, context, plumed_input, data={}):
        super().__init__(context)
        assert 'PRINT' not in plumed_input
        components = parse_plumed_input(plumed_input)
        assert len(components) > 0
        for c in components:
            assert ',' not in c[1] # require 1D bias
        #assert len(set([c[1] for c in components])) == 1 # single CV
        self.components   = components
        self.plumed_input = plumed_input

        # initialize data future for each component
        self.data_futures = OrderedDict()
        for key, value in data.items():
            assert key in self.keys
            if type(value) == str:
                path_new = _new_file(context.path, key + '_', '.txt')
                with open(path_new, 'w') as f:
                    f.write(value)
                self.data_futures[key] = File(path_new)
            else:
                assert (isinstance(value, DataFuture) or isinstance(value, File))
                self.data_futures[key] = value
        for key in self.keys:
            if (key not in self.data_futures.keys()) and (key in PlumedBias.keys_with_future):
                assert key != 'EXTERNAL' # has to be initialized by user
                self.data_futures[key] = File(_new_file(context.path, key + '_', '.txt'))
        if 'METAD' in self.keys:
            self.data_futures.move_to_end('METAD', last=False)

    def evaluate(self, dataset, cv):
        assert cv in [c[1] for c in self.components]
        plumed_input = self.prepare_input()
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if 'ARG=' in line:
                if not (cv == line.split('ARG=')[1].split()[0]):
                    lines[i] = '\n'
        for i, line in enumerate(lines):
            if 'METAD' in line.split():
                line_before = line.split('PACE=')[0]
                line_after  = line.split('PACE=')[1].split()[1:]
                pace = 2147483647 # some random high prime number
                lines[i] = line_before + 'PACE={} '.format(pace) + ' '.join(line_after)
        plumed_input = '\n'.join(lines)
        return self.context.apps(PlumedBias, 'evaluate')(
                plumed_input,
                cv,
                inputs=[dataset.data_future] + self.futures,
                )

    def prepare_input(self):
        plumed_input = str(self.plumed_input)
        for key in self.keys:
            if key in ['METAD', 'EXTERNAL']: # keys for which path needs to be set
                plumed_input = set_path_in_plumed(
                        plumed_input,
                        key,
                        self.data_futures[key].filepath,
                        )
        if 'METAD' in self.keys: # necessary to print hills properly
            plumed_input = 'RESTART\n' + plumed_input
            plumed_input += '\nFLUSH STRIDE=1' # has to come before PRINT?!
        return plumed_input

    def copy(self):
        new_futures = OrderedDict()
        for key, future in self.data_futures.items():
            new_futures[key] = copy_data_future(
                    inputs=[future],
                    outputs=[File(_new_file(self.context.path, 'bias_', '.txt'))],
                    ).outputs[0]
        return PlumedBias(
                self.context,
                self.plumed_input,
                data_futures=new_futures,
                )

    def adjust_restraint(self, cv, kappa, center):
        plumed_input = str(self.plumed_input)
        lines = plumed_input.split('\n')
        found = False
        for i, line in enumerate(lines):
            if 'RESTRAINT' in line.split():
                if 'ARG={}'.format(cv) in line.split():
                    assert not found
                    line_ = line
                    line_before = line_.split('KAPPA=')[0]
                    line_after  = line_.split('KAPPA=')[1].split()[1:]
                    line_ = line_before + 'KAPPA={} '.format(kappa) + ' '.join(line_after)
                    line_before = line_.split('AT=')[0]
                    line_after  = line_.split('AT=')[1].split()[1:]
                    line_ = line_before + 'AT={} '.format(center) + ' '.join(line_after)
                    lines[i] = line_
                    found = True
        assert found
        self.plumed_input = '\n'.join(lines)

    def extract_states(self, dataset, cv, cv_min, cv_max, cv_step, slack=0.1):
        assert cv in [c[1] for c in self.components]
        plumed_input = self.prepare_input()
        lines = plumed_input.split('\n')
        for i, line in enumerate(lines):
            if 'ARG=' in line:
                if not (cv == line.split('ARG=')[1].split()[0]):
                    lines[i] = '\n'
        for i, line in enumerate(lines):
            if 'METAD' in line.split():
                line_before = line.split('PACE=')[0]
                line_after  = line.split('PACE=')[1].split()[1:]
                pace = 2147483647 # some random high prime number
                lines[i] = line_before + 'PACE={} '.format(pace) + ' '.join(line_after)
        plumed_input = '\n'.join(lines)
        indices = self.context.apps(PlumedBias, 'find_states')( # is future!
                plumed_input,
                cv,
                cv_min,
                cv_max,
                cv_step,
                slack,
                inputs=[dataset.data_future] + self.futures,
                )
        return dataset[indices]

    def save(self, path, require_done=True):
        path = Path(path)
        assert path.is_dir()
        path_input = path / 'plumed_input.txt'
        input_future = save_txt(
                self.plumed_input,
                outputs=[File(str(path_input))],
                ).outputs[0]
        data_futures = {}
        for key in self.data_futures.keys():
            path_key = path / (key + '.txt')
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
    def load(cls, context, path):
        path = Path(path)
        assert path.is_dir()
        path_input = path / 'plumed_input.txt'
        assert path_input.is_file()
        with open(path_input, 'r') as f:
            plumed_input = f.read()
        data = {}
        for key in cls.keys_with_future:
            path_key = path / (key + '.txt')
            if path_key.is_file():
                with open(path_key, 'r') as f:
                    data[key] = f.read()
        return cls(context, plumed_input, data=data)

    @property
    def keys(self):
        keys = sorted([c[0] for c in self.components])
        assert len(set(keys)) == len(keys) # keys should be unique!
        if 'METAD' in keys:
            keys.remove('METAD')
            return ['METAD'] + keys
        else:
            return keys

    @property
    def futures(self):
        return [value for _, value in self.data_futures.items()] # MTD first

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ModelExecutionDefinition].executor_label
        app_evaluate = python_app(evaluate_bias, executors=[executor_label])
        context.register_app(cls, 'evaluate', app_evaluate)
        app_find = python_app(find_states_in_data, executors=[executor_label])
        context.register_app(cls, 'find_states', app_find)
