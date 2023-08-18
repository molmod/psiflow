from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
import logging
import copy

from ase.data import atomic_numbers

import parsl
from parsl.app.app import python_app, bash_app, join_app
from parsl.executors import WorkQueueExecutor
from parsl.data_provider.files import File

import psiflow
from psiflow.data import NullState
from psiflow.reference import BaseReference
from psiflow.utils import get_active_executor, copy_app_future


logger = logging.getLogger(__name__) # logging per module


def write_nwchem_in(path_input, atoms, properties=None, echo=False, **params):
    """Adopted from ase.io.nwchem.nwwriter

    forces perm and scratch dirs to their defaults

    """
    import os
    from copy import deepcopy
    from ase.io.nwchem.nwwriter import _get_kpts, _get_theory, _xc_conv, \
            _update_mult, _get_geom, _get_basis, _get_other, _get_set, \
            _get_bandpath
    params = deepcopy(params)
    if properties is None:
        properties = ['energy']
    if 'stress' in properties:
        if 'set' not in params:
            params['set'] = dict()
        params['set']['includestress'] = True
    task = params.get('task')
    if task is None:
        if 'stress' in properties or 'forces' in properties:
            task = 'gradient'
        else:
            task = 'energy'
    params = _get_kpts(atoms, **params)
    theory = _get_theory(**params)
    params['theory'] = theory
    xc = params.get('xc')
    if 'xc' in params:
        xc = _xc_conv.get(params['xc'].lower(), params['xc'])
        if theory in ['dft', 'tddft']:
            if 'dft' not in params:
                params['dft'] = dict()
            params['dft']['xc'] = xc
        elif theory in ['pspw', 'band', 'paw']:
            if 'nwpw' not in params:
                params['nwpw'] = dict()
            params['nwpw']['xc'] = xc
    magmom_tot = int(atoms.get_initial_magnetic_moments().sum())
    params = _update_mult(magmom_tot, **params)
    label = params.get('label', 'nwchem')
    perm = os.path.abspath(params.pop('perm', label))
    scratch = os.path.abspath(params.pop('scratch', label))
    restart_kw = params.get('restart_kw', 'start')
    if restart_kw not in ('start', 'restart'):
        raise ValueError("Unrecognised restart keyword: {}!"
                         .format(restart_kw))
    short_label = label.rsplit('/', 1)[-1]
    if echo:
        out = ['echo']
    else:
        out = []
    out.extend(['title "{}"'.format(short_label),
                #'permanent_dir {}'.format(perm),
                #'scratch_dir {}'.format(scratch),
                '{} {}'.format(restart_kw, short_label),
                '\n'.join(_get_geom(atoms, **params)),
                '\n'.join(_get_basis(**params)),
                '\n'.join(_get_other(**params)),
                '\n'.join(_get_set(**params.get('set', dict()))),
                'task {} {}'.format(theory, task),
                '\n'.join(_get_bandpath(params.get('bandpath', None)))])
    content = '\n\n'.join(out)
    with open(path_input, 'w') as f:
        f.write(content)
    return content


def nwchem_singlepoint_pre(
        atoms: FlowAtoms,
        calculator_kwargs: dict,
        properties: tuple[str, str],
        nwchem_command: str,
        omp_num_threads: int,
        stdout: str = '',
        stderr: str = '',
        walltime: int = 0,
        parsl_resource_specification: Optional[dict] = None,
        ):
    import yaml
    import tempfile
    import shutil
    from ase import io
    from psiflow.reference._nwchem import write_nwchem_in
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    path_input = tmp.name # dummy input file
    calculator_kwargs['perm'] = '/tmp'
    calculator_kwargs['scratch'] = '/tmp'
    write_nwchem_in(
            path_input,
            atoms,
            properties=properties,
            **calculator_kwargs,
            )
    with open(path_input, 'r') as f:
        input_str = f.read()
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_write = 'echo "{}" > nwchem.nwi;'.format(input_str)
    command_mkdir = 'mkdir nwchem;'
    command_list = [
            command_tmp,
            command_cd,
            command_write,
            command_mkdir,
            'timeout -k 5 {}s'.format(max(walltime - 20, 0)),
            nwchem_command + ' nwchem.nwi || true',
            ]
    return ' '.join(command_list)

def nwchem_singlepoint_post(
        atoms: FlowAtoms,
        inputs: list[File] = [],
        ) -> FlowAtoms:
    from ase import io
    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    try:
        results = io.read(inputs[0], format='nwchem-out')
        atoms.info['energy'] = results.calc.results['energy']
        atoms.reference_status = True
    except Exception as e:
        print(e)
        atoms.reference_status = False
    try: # OK if no forces present
        atoms.arrays['forces'] = results.calc.results['forces']
    except KeyError: # when only energies were requested
        with open(atoms.reference_stderr, 'r') as f:
            content = f.read()
        assert 'task' in content
        assert not 'dft gradient' in content
        atoms.arrays['forces'] = None
    return atoms


class NWChemReference(BaseReference):
    required_files = []

    def __init__(self, properties=('energy', 'forces'), **calculator_kwargs):
        self.properties = properties
        self.calculator_kwargs = calculator_kwargs
        super().__init__()

    def get_single_atom_references(self, element):
        number = atomic_numbers[element]
        references = []
        for mult in range(1, 16):
            config = {'mult': mult}
            calculator_kwargs = copy.deepcopy(self.calculator_kwargs)
            if number % 2 == 0 and mult % 2 == 0:
                continue # not 2N + 1 is never even
            if mult == 1 and number % 2 == 1:
                continue # nwchem errors when mult = 1 for odd number of electrons
            if mult - 1 > number:
                continue # max S = 2 * (N * 1/2) + 1
            if 'dft' in calculator_kwargs:
                calculator_kwargs['dft']['mult'] = mult
            else:
                raise NotImplementedError
            reference = self.__class__(
                    properties=('energy'),
                    **calculator_kwargs,
                    )
            references.append((config, reference))
        return references

    @property
    def parameters(self):
        return {
                'calculator_kwargs': self.calculator_kwargs,
                'properties': self.properties,
                }

    @classmethod
    def create_apps(cls):
        context = psiflow.context()
        definition = context[cls]
        label       = definition.name()
        mpi_command = definition.mpi_command
        ncores      = definition.cores_per_worker
        walltime    = definition.max_walltime
        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_specification = definition.generate_parsl_resource_specification()
        else:
            resource_specification = {}

        # parse full command
        omp_num_threads = 1
        command = ''
        command += mpi_command(ncores)
        command += ' '
        command += 'nwchem'
        
        singlepoint_pre = bash_app(
                nwchem_singlepoint_pre,
                executors=[label],
                )
        singlepoint_post = python_app(
                nwchem_singlepoint_post,
                executors=['Default'],
                )

        @join_app
        def singlepoint_wrapped(
                atoms,
                parameters,
                file_names,
                inputs=[]
                ):
            assert len(file_names) == 0
            if atoms == NullState:
                return copy_app_future(NullState)
            else:
                pre = singlepoint_pre(
                        atoms,
                        parameters['calculator_kwargs'],
                        parameters['properties'],
                        command,
                        omp_num_threads,
                        stdout=parsl.AUTO_LOGNAME,
                        stderr=parsl.AUTO_LOGNAME,
                        walltime=60 * walltime, # killed after walltime - 10s
                        parsl_resource_specification=resource_specification,
                        )
                return singlepoint_post(
                        atoms=atoms,
                        inputs=[pre.stdout, pre.stderr, pre], # wait for bash app
                        )
        context.register_app(cls, 'evaluate_single', singlepoint_wrapped)
        super(NWChemReference, cls).create_apps()
