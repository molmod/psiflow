from __future__ import annotations  # necessary for type-guarding class methods

import logging
from typing import Optional

import numpy as np
import parsl
from ase import Atoms
from ase.data import atomic_numbers
from parsl.app.app import bash_app, join_app, python_app
from parsl.data_provider.files import File
from parsl.executors import WorkQueueExecutor

import psiflow
from psiflow.data import FlowAtoms, NullState
from psiflow.reference.base import BaseReference
from psiflow.utils import copy_app_future, get_active_executor

logger = logging.getLogger(__name__)  # logging per module


def atoms_to_molecule(ase_atoms, basis, spin):
    from pyscf import gto
    from pyscf.pbc import gto as pbcgto

    atom_symbols = ase_atoms.get_chemical_symbols()
    atom_coords = ase_atoms.get_positions()
    atom_spec = [
        (symbol, tuple(coord)) for symbol, coord in zip(atom_symbols, atom_coords)
    ]

    if ase_atoms.get_pbc().any():  # Periodic boundary conditions
        cell_params = ase_atoms.get_cell()
        pyscf_cell = pbcgto.Cell()
        pyscf_cell.atom = atom_spec
        pyscf_cell.basis = basis
        pyscf_cell.spin = spin
        pyscf_cell.a = cell_params
        pyscf_cell.build()
        return pyscf_cell
    else:  # Non-periodic (molecular)
        pyscf_mol = gto.Mole()
        pyscf_mol.atom = atom_spec
        pyscf_mol.basis = basis
        pyscf_mol.spin = spin
        pyscf_mol.verbose = 5
        pyscf_mol.build()
        return pyscf_mol


def serialize_atoms(atoms):
    atoms_str = "dict(symbols={}, positions={}, cell={}, pbc={})".format(
        atoms.get_chemical_symbols(),
        atoms.get_positions().tolist(),
        atoms.get_cell().tolist(),
        atoms.get_pbc().tolist(),
    )
    return atoms_str


def deserialize_atoms(atoms_dict):
    return Atoms(
        symbols=atoms_dict["symbols"],
        positions=np.array(atoms_dict["positions"]),  # Convert list back to numpy array
        cell=np.array(atoms_dict["cell"]),
        pbc=atoms_dict["pbc"],
    )


def generate_script(state, routine, basis, spin):
    # print 'energy' and 'forces' variables
    routine = routine.strip()
    routine += """
print('total energy = {}'.format(energy * Ha))
print('total forces = ')
for force in forces:
    print(*(force * Ha / Bohr))
"""
    lines = routine.split("\n")  # indent entire routine
    for i in range(len(lines)):
        lines[i] = "    " + lines[i]
    routine = "\n".join(lines)

    script = """
import scipy # avoids weird circular import inside pyscf
from ase.units import Ha, Bohr
from psiflow.reference._pyscf import deserialize_atoms, atoms_to_molecule

def main(molecule):
{}


""".format(
        routine
    )
    script += """
if __name__ == '__main__':
    atoms_dict = {}
    atoms = deserialize_atoms(atoms_dict)
    molecule = atoms_to_molecule(
            atoms,
            basis='{}',
            spin={},
            )
    main(molecule)
""".format(
        serialize_atoms(state).strip(),
        basis,
        spin,
    )
    return script


def parse_energy_forces(stdout):
    energy = None
    forces_str = None
    lines = stdout.split("\n")
    for i, line in enumerate(lines[::-1]):  # start from back!
        if energy is None and "total energy = " in line:
            energy = float(line.split("total energy = ")[1])
        if forces_str is None and "total forces =" in line:
            forces_str = "\n".join(lines[-i:])
    assert energy is not None
    assert forces_str is not None
    rows = forces_str.strip().split("\n")
    nrows = len(rows)
    ncols = len(rows[0].split())
    assert ncols == 3
    forces = np.fromstring("\n".join(rows), sep=" ", dtype=float)
    return energy, np.reshape(forces, (nrows, ncols))


def pyscf_singlepoint_pre(
    atoms: FlowAtoms,
    omp_num_threads: int,
    stdout: str = "",
    stderr: str = "",
    walltime: int = 0,
    parsl_resource_specification: Optional[dict] = None,
    **parameters,
) -> str:
    from psiflow.reference._pyscf import generate_script

    script = generate_script(atoms, **parameters)
    command_tmp = 'mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t "mytmpdir");'
    command_cd = "cd $mytmpdir;"
    command_write = 'echo "{}" > generated.py;'.format(script)
    command_list = [
        command_tmp,
        command_cd,
        command_write,
        "export OMP_NUM_THREADS={};".format(omp_num_threads),
        "export OMP_PROC_BIND=true;",
        "export KMP_AFFINITY=granularity=fine,compact,1,0;"
        "timeout -s 9 {}s python generated.py || true".format(max(walltime - 2, 0)),
    ]
    return " ".join(command_list)


def pyscf_singlepoint_post(
    atoms: FlowAtoms,
    inputs: list[File] = [],
) -> FlowAtoms:
    from psiflow.reference._pyscf import parse_energy_forces

    atoms.reference_stdout = inputs[0]
    atoms.reference_stderr = inputs[1]
    with open(atoms.reference_stdout, "r") as f:
        content = f.read()
    try:
        energy, forces = parse_energy_forces(content)
        assert forces.shape == atoms.positions.shape
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = forces
        atoms.reference_status = True
    except Exception:
        atoms.reference_status = False
    return atoms


class PySCFReference(BaseReference):
    def __init__(self, routine, basis, spin):
        assert (
            "energy = " in routine
        ), "define the total energy (in Ha) in your pyscf routine"
        assert (
            "forces = " in routine
        ), "define the forces (in Ha/Bohr) in your pyscf routine"
        assert "pyscf" in routine, "put all necessary imports inside the routine!"
        self.routine = routine
        self.basis = basis
        self.spin = spin
        super().__init__()

    def get_single_atom_references(self, element):
        number = atomic_numbers[element]
        references = []
        for spin in range(15):
            config = {"spin": spin}
            mult = spin + 1
            if number % 2 == 0 and mult % 2 == 0:
                continue
            if mult == 1 and number % 2 == 1:
                continue
            if mult - 1 > number:
                continue
            parameters = self.parameters
            parameters["spin"] = spin
            reference = self.__class__(**parameters)
            references.append((config, reference))
        return references

    @property
    def parameters(self):
        return {
            "routine": self.routine,
            "basis": self.basis,
            "spin": self.spin,
        }

    @classmethod
    def create_apps(cls):
        context = psiflow.context()
        definition = context[cls]
        label = definition.name()
        ncores = definition.cores_per_worker
        walltime = definition.max_walltime

        if isinstance(get_active_executor(label), WorkQueueExecutor):
            resource_specification = definition.generate_parsl_resource_specification()
        else:
            resource_specification = {}

        singlepoint_pre = bash_app(
            pyscf_singlepoint_pre,
            executors=[label],
        )
        singlepoint_post = python_app(
            pyscf_singlepoint_post,
            executors=["default_threads"],
        )

        @join_app
        def singlepoint_wrapped(atoms, parameters):
            if atoms == NullState:
                return copy_app_future(NullState)
            else:
                pre = singlepoint_pre(
                    atoms,
                    omp_num_threads=ncores,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    walltime=60 * walltime,  # killed after walltime - 10s
                    parsl_resource_specification=resource_specification,
                    **parameters,
                )
                return singlepoint_post(
                    atoms=atoms,
                    inputs=[pre.stdout, pre.stderr, pre],  # wait for bash app
                )

        context.register_app(cls, "evaluate_single", singlepoint_wrapped)
        super(PySCFReference, cls).create_apps()
