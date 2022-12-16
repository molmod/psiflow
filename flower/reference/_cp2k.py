from dataclasses import dataclass

from parsl.app.app import python_app

from flower.execution import ReferenceExecutionDefinition
from .base import BaseReference


def insert_filepaths_in_input(cp2k_input, filepaths):
    from pymatgen.io.cp2k.inputs import Cp2kInput, Keyword, KeywordList
    inp = Cp2kInput.from_string(cp2k_input)
    for key, path in filepaths.items():
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


def insert_atoms_in_input(cp2k_input, atoms):
    from pymatgen.io.cp2k.inputs import Cp2kInput, Cell, Coord
    from pymatgen.core import Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    structure = AseAtomsAdaptor.get_structure(atoms)
    lattice = Lattice(atoms.get_cell())

    inp = Cp2kInput.from_string(cp2k_input)
    if not 'SUBSYS' in inp['FORCE_EVAL'].subsections.keys():
        raise ValueError('No subsystem present in cp2k input: {}'.format(cp2k_input))
    inp['FORCE_EVAL']['SUBSYS'].insert(Coord(structure))
    inp['FORCE_EVAL']['SUBSYS'].insert(Cell(lattice))
    return str(inp)


def regularize_input(cp2k_input):
    """Ensures forces and stress are printed; removes topology/cell info"""
    from pymatgen.io.cp2k.inputs import Cp2kInput
    inp = Cp2kInput.from_string(cp2k_input)
    inp.update({'FORCE_EVAL': {'SUBSYS': {'CELL': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'TOPOLOGY': {}}}})
    inp.update({'FORCE_EVAL': {'SUBSYS': {'COORD': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'FORCES': {}}}})
    inp.update({'FORCE_EVAL': {'PRINT': {'STRESS_TENSOR': {}}}})
    return str(inp)


def set_global_section(cp2k_input):
    from pymatgen.io.cp2k.inputs import Cp2kInput, Global
    inp = Cp2kInput.from_string(cp2k_input)
    inp.subsections['GLOBAL'] = Global(project_name='_electron')
    return str(inp)


def cp2k_singlepoint(
        atoms,
        parameters,
        command,
        walltime,
        inputs=[],
        outputs=[],
        ):
    import tempfile
    import subprocess
    import ase
    import glob
    import os
    import shlex
    import parsl
    from pathlib import Path
    import numpy as np
    from ase.units import Hartree, Bohr
    from pymatgen.io.cp2k.outputs import Cp2kOutput
    from flower.reference._cp2k import insert_filepaths_in_input, \
            insert_atoms_in_input, set_global_section

    command_list = [command]
    with tempfile.TemporaryDirectory() as tmpdir:
        # write data files as required by cp2k
        filepaths = {}
        for key, content in parameters.cp2k_data.items():
            filepaths[key] = Path(tmpdir) / key
            with open(filepaths[key], 'w') as f:
                f.write(content)
        cp2k_input = insert_filepaths_in_input(
                parameters.cp2k_input,
                filepaths,
                )
        cp2k_input = regularize_input(cp2k_input) # before insert_atoms_in_input
        cp2k_input = insert_atoms_in_input(
                cp2k_input,
                atoms,
                )
        cp2k_input = set_global_section(cp2k_input)
        path_input  = Path(tmpdir) / 'cp2k_input.txt'
        with open(Path(tmpdir) / 'cp2k_input.txt', 'w') as f:
            f.write(cp2k_input)
        command_list.append(' -i {}'.format(path_input))
        os.environ['OMP_NUM_THREADS'] = '1'
        try:
            result = subprocess.run(
                    shlex.split(' '.join(command_list)), # proper splitting
                    env=dict(os.environ),
                    shell=False, # to be able to use timeout
                    capture_output=True,
                    text=True,
                    timeout=walltime,
                    )
            stdout = result.stdout
            stderr = result.stderr
            timeout = False
            returncode = result.returncode
            success = (returncode == 0)
        except subprocess.CalledProcessError as e:
            stdout = result.stdout
            stderr = result.stderr
            timeout = False
            returncode = 1
            success = False
            #print(e)
        except parsl.app.errors.AppTimeout as e: # subprocess.TimeoutExpired
            #stdout = e.stdout.decode('utf-8') # no result variable in this case
            #stderr = e.stderr
            stdout = ''
            stderr = 'subprocess walltime ({}s) reached'.format(walltime)
            timeout = True
            returncode = 1
            success = False
        print('success: {}\treturncode: {}\ttimeout: {}'.format(success, returncode, timeout))
        atoms.evaluation_log = stdout
        if success:
            atoms.evaluation_flag = 'success'
            with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmp:
                tmp.write(atoms.evaluation_log)
            out = Cp2kOutput(tmp.name)
            out.parse_energies()
            out.parse_forces()
            out.parse_stresses()
            energy = out.data['total_energy'][0] # already in eV
            forces = np.array(out.data['forces'][0]) * (Hartree / Bohr) # to eV/A
            stress = np.array(out.data['stress_tensor'][0]) * 1000 # to MPa
            atoms.info['energy'] = energy
            atoms.info['stress'] = stress
            atoms.arrays['forces'] = forces
            for file in glob.glob('_electron-RESTART.wfn*'):
                os.remove(file) # include .wfn.bak-
        else:
            atoms.evaluation_flag = 'failed'
            atoms.evaluation_log += '\n\n STDERR\n' + stderr
            # remove properties keys in atoms if present
            atoms.info.pop('energy', None)
            atoms.info.pop('stress', None)
            atoms.arrays.pop('forces', None)
        return atoms


@dataclass
class CP2KParameters:
    cp2k_input : str
    cp2k_data  : dict


class CP2KReference(BaseReference):
    """CP2K Reference"""
    parameters_cls = CP2KParameters

    def __init__(self, context, cp2k_input='', cp2k_data={}):
        """Constructor

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
        super().__init__(context, cp2k_input=cp2k_input, cp2k_data=cp2k_data)

    @classmethod
    def create_apps(cls, context):
        executor_label = context[ReferenceExecutionDefinition].executor_label
        ncores = context[ReferenceExecutionDefinition].ncores
        mpi_command = context[ReferenceExecutionDefinition].mpi_command
        cp2k_exec = context[ReferenceExecutionDefinition].cp2k_exec
        walltime = context[ReferenceExecutionDefinition].walltime

        # parse full command
        command = ''
        if mpi_command is not None:
            command += mpi_command(ncores)
        command += ' '
        command += cp2k_exec

        # convert walltime into seconds
        hms = walltime.split(':')
        _walltime = float(hms[2]) + 60 * float(hms[1]) + 3600 * float(hms[0])
        singlepoint_unwrapped = python_app(
                cp2k_singlepoint,
                executors=[executor_label],
                )
        def singlepoint_wrapped(atoms, parameters, inputs=[], outputs=[]):
            assert len(outputs) == 0
            return singlepoint_unwrapped(
                    atoms=atoms,
                    parameters=parameters,
                    command=command,
                    walltime=_walltime,
                    inputs=inputs,
                    outputs=[],
                    )
        context.register_app(cls, 'evaluate_single', singlepoint_wrapped)
        super(CP2KReference, cls).create_apps(context)
