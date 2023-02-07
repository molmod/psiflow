from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union
import typeguard
from dataclasses import dataclass
import tempfile
import shutil

import parsl
from parsl.executors import WorkQueueExecutor
from parsl.app.app import python_app, bash_app
from parsl.dataflow.memoization import id_for_memo
from parsl.data_provider.files import File

from psiflow.execution import ReferenceEvaluationExecution
from psiflow.data import FlowAtoms
from psiflow.utils import get_active_executor
from .base import BaseReference


#@typeguard.typechecked
def insert_filepaths_in_input(
        cp2k_input: str,
        files: dict[str, Union[str, list[str]]]) -> str:
    from pymatgen.io.cp2k.inputs import Cp2kInput, Keyword, KeywordList
    inp = Cp2kInput.from_string(cp2k_input)
    for name, path in files.items():
        if name == 'basis_set':
            key = 'BASIS_SET_FILE_NAME'
        elif name == 'potential':
            key = 'POTENTIAL_FILE_NAME'
        elif name == 'dftd3':
            key = 'PARAMETER_FILE_NAME'

        if isinstance(path, list): # set as KeywordList
            keywords = []
            for _path in path:
                keywords.append(Keyword(key, _path, repeats=True))
            to_add = KeywordList(keywords)
        else:
            to_add = Keyword(key, path, repeats=False)
        if key == 'BASIS_SET_FILE_NAME':
            inp.update({'FORCE_EVAL': {'DFT': {key: to_add}}}, strict=True)
        elif key == 'POTENTIAL_FILE_NAME':
            inp.update({'FORCE_EVAL': {'DFT': {key: to_add}}}, strict=True)
        elif key == 'PARAMETER_FILE_NAME':
            inp.update(
                    {'FORCE_EVAL': {'DFT': {'XC': {'VDW_POTENTIAL': {'PAIR_POTENTIAL': {key: to_add}}}}}},
                    strict=True,
                    )
        else:
            raise ValueError('File key {} not recognized'.format(key))
    return str(inp)


@typeguard.typechecked
def insert_atoms_in_input(cp2k_input: str, atoms: FlowAtoms) -> str:
    from ase.data import chemical_symbols
    from pymatgen.io.cp2k.inputs import Cp2kInput
    inp = Cp2kInput.from_string(cp2k_input)
    if not 'SUBSYS' in inp['FORCE_EVAL'].subsections.keys():
        raise ValueError('No subsystem present in cp2k input: {}'.format(cp2k_input))
    try:
        del inp['FORCE_EVAL']['SUBSYS']['TOPOLOGY'] # remove just to be safety
    except KeyError:
        pass
    try:
        del inp['FORCE_EVAL']['SUBSYS']['COORD'] # remove just to be safety
    except KeyError:
        pass
    try:
        del inp['FORCE_EVAL']['SUBSYS']['CELL'] # remove just to be safety
    except KeyError:
        pass
    cp2k_input = str(inp)

    for line in cp2k_input.splitlines():
        assert '&COORD' not in line
        assert '&TOPOLOGY' not in line
        assert '&CELL' not in line

    # insert atomic positions
    atoms_str = ''
    for i in range(len(atoms)):
        n   = atoms.numbers[i]
        pos = [str(c) for c in atoms.positions[i]]
        atoms_str += str(chemical_symbols[n]) + ' ' + ' '.join(pos) + '\n'
    atoms_str += '\n'
    for i, line in enumerate(cp2k_input.splitlines()):
        if tuple(line.split()) == ('&END', 'SUBSYS'): # insert before here
            break
    assert i != len(cp2k_input.splitlines()) - 1
    new_input = '\n'.join(cp2k_input.splitlines()[:i]) + '\n'
    new_input += '&COORD\n' + atoms_str + '&END COORD\n'
    new_input += '\n'.join(cp2k_input.splitlines()[i:])
    cp2k_input = new_input

    # insert box vectors
    cell_str = ''
    for index, name in [(0, 'A'), (1, 'B'), (2, 'C')]:
        vector = [str(c) for c in atoms.cell[index]]
        cell_str += name + ' ' + ' '.join(vector) + '\n'
    cell_str += '\n'
    for i, line in enumerate(cp2k_input.splitlines()):
        if tuple(line.split()) == ('&END', 'SUBSYS'): # insert before here
            break
    new_input = '\n'.join(cp2k_input.splitlines()[:i]) + '\n'
    new_input += '&CELL\n' + cell_str + '&END CELL\n'
    new_input += '\n'.join(cp2k_input.splitlines()[i:])
    cp2k_input = new_input

    return cp2k_input


@typeguard.typechecked
def regularize_input(cp2k_input: str) -> str:
    """Ensures forces and stress are printed; removes topology/cell info"""
    from pymatgen.io.cp2k.inputs import Cp2kInput
    inp = Cp2kInput.from_string(cp2k_input)
    inp.update({'FORCE_EVAL': {'SUBSYS': {'CELL': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'TOPOLOGY': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'COORD': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'FORCES': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'STRESS_TENSOR': {}}}})
    return str(inp)


@typeguard.typechecked
def set_global_section(cp2k_input: str) -> str:
    from pymatgen.io.cp2k.inputs import Cp2kInput, Global
    inp = Cp2kInput.from_string(cp2k_input)
    inp.subsections['GLOBAL'] = Global(project_name='cp2k_project')
    return str(inp)


# typeguarding not compatible with parsl WQEX for some reason
def cp2k_singlepoint_pre(
        atoms: FlowAtoms,
        parameters: CP2KParameters,
        cp2k_command: str,
        file_names: list[str],
        omp_num_threads: int,
        walltime: int = 0,
        inputs: list = [],
        stdout: str = '',
        stderr: str = '',
        parsl_resource_specification: Optional[dict] = None,
        ):
    import tempfile
    import glob
    from pathlib import Path
    import numpy as np
    from psiflow.reference._cp2k import insert_filepaths_in_input, \
            insert_atoms_in_input, set_global_section
    filepaths = {} # cp2k cannot deal with long filenames; copy into local dir
    for name, file in zip(file_names, inputs):
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        tmp.close()
        shutil.copyfile(file.filepath, tmp.name)
        filepaths[name] = tmp.name
    cp2k_input = insert_filepaths_in_input(
            parameters.cp2k_input,
            filepaths,
            )
    cp2k_input = regularize_input(cp2k_input) # before insert_atoms_in_input
    cp2k_input = set_global_section(cp2k_input)
    cp2k_input = insert_atoms_in_input(
            cp2k_input,
            atoms,
            )
    # see https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd  = 'cd $mytmpdir;'
    command_write = 'echo "{}" > cp2k.inp;'.format(cp2k_input)
    command_list = [
            command_tmp,
            command_cd,
            command_write,
            'export OMP_NUM_THREADS={};'.format(omp_num_threads),
            'timeout {}s'.format(max(walltime - 10, 0)), # some time is spent on copying
            cp2k_command,
            '-i cp2k.inp',
            ' || true',
            ]
    return ' '.join(command_list)


def cp2k_singlepoint_post(
        atoms: FlowAtoms,
        inputs: list[File] = [],
        ) -> FlowAtoms:
    import numpy as np
    from ase.units import Hartree, Bohr, Pascal
    from pymatgen.io.cp2k.outputs import Cp2kOutput
    with open(inputs[0], 'r') as f:
        stdout = f.read()
    with open(inputs[1], 'r') as f:
        stderr = f.read()
    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    try:
        out = Cp2kOutput(inputs[0])
        out.parse_energies()
        out.parse_forces()
        out.parse_stresses()
        energy_ = out.data['total_energy'][0] # already in eV
        forces_ = np.array(out.data['forces'][0]) * (Hartree / Bohr) # to eV/A
        stress_ = np.array(out.data['stress_tensor'][0]) * (1e9 * Pascal)
        atoms.info['energy'] = energy_
        atoms.info['stress'] = stress_
        atoms.arrays['forces'] = forces_
        atoms.reference_status = True
    except:
        atoms.reference_status = False
    return atoms


@dataclass
class CP2KParameters:
    cp2k_input : str


@id_for_memo.register(CP2KParameters)
def id_for_memo_cp2k_parameters(parameters: CP2KParameters, output_ref=False):
    assert not output_ref
    b1 = id_for_memo(parameters.cp2k_input, output_ref=output_ref)
    return b1


class CP2KReference(BaseReference):
    """CP2K Reference

    Arguments
    ---------

    cp2k_input : str
        string representation of the cp2k input file.

    cp2k_data : dict
        dictionary with data required during the calculation. E.g. basis
        sets, pseudopotentials, ...
        They are written to the local execution directory in order to make
        them available to the cp2k executable.
        The keys of the dictionary correspond to the capitalized keys in
        the cp2k input (e.g. BASIS_SET_FILE_NAME)

    """
    execution_types = set([ReferenceEvaluationExecution])
    parameters_cls = CP2KParameters
    required_files = [
            'basis_set',
            'potential',
            'dftd3',
            ]

    @classmethod
    def create_apps(cls, context):
        execution  = context[cls][0]
        label       = execution.executor
        device      = execution.device
        mpi_command = execution.mpi_command
        cp2k_exec   = execution.cp2k_exec
        ncores      = execution.ncores
        walltime    = execution.walltime
        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_specification = execution.generate_parsl_resource_specification()
        else:
            resource_specification = {}

        omp_num_threads = execution.omp_num_threads
        assert ncores % execution.omp_num_threads == 0
        mpi_num_procs = ncores // execution.omp_num_threads
        assert device == 'cpu' # cuda not supported at the moment

        # parse full command
        command = ''
        if mpi_command is not None:
            command += mpi_command(mpi_num_procs)
        command += ' '
        command += cp2k_exec

        singlepoint_pre = bash_app(
                cp2k_singlepoint_pre,
                executors=[label],
                cache=False,
                )
        singlepoint_post = python_app(
                cp2k_singlepoint_post,
                executors=[label],
                cache=False,
                )
        def singlepoint_wrapped(
                atoms,
                parameters,
                file_names,
                inputs=[],
                ):
            assert len(file_names) == len(inputs)
            for name in cls.required_files:
                assert name in file_names
            pre = singlepoint_pre(
                    atoms,
                    parameters,
                    command,
                    file_names,
                    omp_num_threads,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    walltime=60 * walltime, # killed after walltime - 10s
                    inputs=inputs, # tmp Files
                    parsl_resource_specification=resource_specification,
                    )
            return singlepoint_post(
                    atoms=atoms,
                    inputs=[pre.stdout, pre.stderr, pre], # wait for bash app
                    )
        context.register_app(cls, 'evaluate_single', singlepoint_wrapped)
        super(CP2KReference, cls).create_apps(context)
