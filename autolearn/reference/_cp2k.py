import glob
import os
import subprocess
import tempfile
import shlex
from pathlib import Path
import numpy as np
import covalent as ct

from pymatgen.io.cp2k.inputs import Cp2kInput, Keyword, KeywordList, Cell, \
        Coord, Global
from pymatgen.io.cp2k.outputs import Cp2kOutput
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice

from autolearn.base import BaseReference


def insert_filepaths_in_input(cp2k_input, filepaths):
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
    structure = AseAtomsAdaptor.get_structure(atoms)
    lattice = Lattice(atoms.get_cell())

    inp = Cp2kInput.from_string(cp2k_input)
    if not 'SUBSYS' in inp['FORCE_EVAL'].subsections.keys():
        raise ValueError('No subsystem present in cp2k input: {}'.format(cp2k_input))
    inp['FORCE_EVAL']['SUBSYS'].insert(Coord(structure))
    inp['FORCE_EVAL']['SUBSYS'].insert(Cell(lattice))
    return str(inp)


def set_global_section(cp2k_input):
    inp = Cp2kInput.from_string(cp2k_input)
    inp.subsections['GLOBAL'] = Global(project_name='_electron')
    return str(inp)


class CP2KReference(BaseReference):
    """CP2K Reference"""

    def __init__(self, cp2k_input, data):
        """Constructor

        Arguments
        ---------

        cp2k_input : str
            string representation of the cp2k input file.

        data : dict
            dictionary with data required during the calculation. E.g. basis
            sets, pseudopotentials, ...
            They are written to the local execution directory in order to make
            them available to the cp2k executable.
            The keys of the dictionary correspond to the capitalized keys in
            the cp2k input (e.g. BASIS_SET_FILE_NAME)

        """
        self.cp2k_input = cp2k_input
        self.data = data

    def evaluate(self, sample, reference_execution):
        ncores   = reference_execution.ncores
        command  = reference_execution.command
        mpi      = reference_execution.mpi
        walltime = reference_execution.walltime
        command_list = []
        if mpi is not None:
            command_list += mpi(ncores)
        command_list.append(command)
        def evaluate_barebones(sample, reference):
            with tempfile.TemporaryDirectory() as tmpdir:
                # write data files as required by cp2k
                filepaths = {}
                for key, content in reference.data.items():
                    filepaths[key] = Path(tmpdir) / key
                    with open(filepaths[key], 'w') as f:
                        f.write(content)
                cp2k_input = insert_filepaths_in_input(
                        reference.cp2k_input,
                        filepaths,
                        )
                cp2k_input = insert_atoms_in_input(
                        cp2k_input,
                        sample.atoms,
                        )
                cp2k_input = set_global_section(cp2k_input)
                path_input  = Path(tmpdir) / 'cp2k_input.txt'
                path_output = Path(tmpdir) / 'cp2k_output.txt'
                with open(Path(tmpdir) / 'cp2k_input.txt', 'w') as f:
                    f.write(cp2k_input)
                command_list.append('-i {}'.format(path_input))
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
                    print(e)
                except subprocess.TimeoutExpired as e:
                    stdout = e.stdout.decode('utf-8') # no result variable in this case
                    stderr = e.stderr
                    timeout = True
                    returncode = 1
                    success = False
                    print(e)
                print('success: {}\treturncode: {}\ttimeout: {}'.format(success, returncode, timeout))
                print(stdout)
                print(stderr)
                if success:
                    with open(path_output, 'w') as f:
                        f.write(stdout)
                    out = Cp2kOutput(path_output)
                    out.parse_energies()
                    out.parse_forces()
                    out.parse_stresses()
                    energy = out.data['total_energy'][0]
                    forces = np.array(out.data['forces'][0])
                    stress = np.array(out.data['stress_tensor'][0])
                    sample.label(
                            energy,
                            forces,
                            stress,
                            log=stdout,
                            )
                    sample.tag('success')
                    for file in glob.glob('_electron-RESTART.wfn*'):
                        os.remove(file) # include .wfn.bak-
                else:
                    sample.tag('error')
                    if timeout:
                        sample.tag('timeout')
                    sample.log = stdout
            return sample
        evaluate_electron = ct.electron(
                evaluate_barebones,
                executor=reference_execution.executor,
                )
        return evaluate_electron(sample, self)
