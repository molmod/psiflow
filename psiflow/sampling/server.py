import argparse
import signal
import xml.etree.ElementTree as ET
import warnings
import subprocess
from pathlib import Path

import ase
import numpy as np
from ase.data import atomic_numbers
from ase.geometry import Cell
from ase.io import read, write
from ase.units import Bohr, Ha, _e, _hbar

from psiflow.geometry import Geometry
from psiflow.sampling.utils import TimeoutException, timeout_handler

NONPERIODIC_CELL = 1000 * np.eye(3)


def parse_checkpoint(file_xml: str | Path) -> list[Geometry]:
    """"""
    from ipi.inputs.system import InputSystem
    from ipi.utils.io.inputs.io_xml import xml_parse_file

    xml_sim = xml_parse_file(open(file_xml)).fields[0][1]  # see Simulation.load_from_xml
    xml_systems = {v.attribs['prefix']: v for k, v in xml_sim.fields if k == 'system'}

    geometries = []
    for name, xml_sys in xml_systems.items():
        sys = InputSystem()
        sys.parse(xml_sys)

        # get structural properties
        beads = sys.beads.fetch()
        natoms, nbeads = beads.natoms, beads.nbeads
        positions = np.asarray(beads.q).reshape(nbeads, natoms, 3).mean(axis=0)  # average over beads
        numbers = np.array([atomic_numbers[s] for s in beads.names])
        cell = np.asarray(sys.cell.fetch().h).T  # transpose to undo i-Pi convention
        geometry = Geometry.from_data(numbers, positions * Bohr, cell * Bohr)

        # get current internal system time
        ensemble_data = sys.ensemble.fetch()
        time = ensemble_data.time * (_hbar / (_e * Ha)) * 1e12
        geometry.order['time'], geometry.order['name'] = time, name
        geometries.append(geometry)

    return geometries


def insert_addresses(input_xml: ET.Element) -> None:
    """Append working directory to socket names"""
    for child in input_xml:
        if child.tag == "ffsocket":
            for child_ in child:
                if child_.tag == "address":
                    address = child_
                    break
            address.text = str(Path.cwd() / address.text.strip())


def insert_data_start(input_xml: ET.Element) -> None:
    # TODO: does this do anything?
    for child in input_xml:
        if child.tag == "system":
            initialize = ET.Element("initialize", nbeads="1")
            initialize.text = "start_INDEX.xyz"

            warnings.warn('"insert_data_start" did something -- investigate')


def anisotropic_barostat_h0(input_xml: ET.Element, data_start: list[ase.Atoms]) -> None:
    """Insert cell reference into barostat"""
    path = "system_template/template/system/motion/dynamics/barostat"
    barostat = input_xml.find(path)
    if barostat is not None and (barostat.attrib["mode"] == "anisotropic"):
        h0 = ET.SubElement(barostat, "h0", shape="(3, 3)", units="angstrom")
        cell = np.array(data_start[0].cell).flatten(order="F")                  # TODO: what if cells are different?
        h0.text = " [ {} ] ".format(" , ".join([str(a) for a in cell]))         # TODO: create_xml_list?

        path = "system_template/template/system/ensemble"
        ensemble = input_xml.find(path)
        assert ensemble is not None
        pressure = ensemble.find("pressure")
        if pressure is not None:
            ensemble.remove(pressure)
            stress = ET.SubElement(ensemble, "stress", units="megapascal")
            stress.text = " [ PRESSURE, 0, 0, 0, PRESSURE, 0, 0, 0, PRESSURE ] "        # TODO: anisotropic?


def run(args):
    from ipi.engine.simulation import Simulation
    from ipi.utils.softexit import softexit

    # prepare starting geometries from context_dir
    data_start = read(args.start_xyz, index=":")
    assert len(data_start) == args.nwalkers
    for i, at in enumerate(data_start):
        if not any(at.pbc):  # set fake large cell for i-PI
            at.pbc = True
            at.cell = Cell(NONPERIODIC_CELL)
        write("start_{}.xyz".format(i), data_start[i])

    with open(args.input_xml, "r") as f:
        input_xml = ET.fromstring(f.read())

    insert_data_start(input_xml)
    insert_addresses(input_xml)
    anisotropic_barostat_h0(input_xml, data_start)
    with open("input.xml", "wb") as f:
        f.write(ET.tostring(input_xml, encoding="utf-8"))

    simulation = Simulation.load_from_xml("input.xml", sockets_prefix="")
    try:
        simulation.run()
        softexit.trigger(status="success", message=" @ SIMULATION: Exiting cleanly.")
    except TimeoutException:
        print("simulation timed out -- killing gracefully")


def cleanup(args):
    from psiflow.data.utils import _write_frames
    print('STARTING CLEANUP')
    with open("input.xml", "r") as f:
        content = f.read()
    if "vibrations" in content:
        # do stuff
        return

    # collect final checkpoint geometries
    states = parse_checkpoint('output.checkpoint')
    for state in states:
        if np.allclose(state.cell, NONPERIODIC_CELL):
            state.cell[:] = 0.0
    _write_frames(*states, outputs=[args.output_xyz])
    print('MOVED CHECKPOINT GEOMETRIES')

    prefix = ''
    if "remd" in content:
        # unshuffle simulation output according to ensemble
        prefix = 'SRT_'
        # TODO: the i-pi version packaged with Psiflow requires .ase extension -- this is fixed in newer versions
        for path in Path.cwd().glob('*.trajectory*.extxyz'):
            path.rename(path.with_suffix('.ase'))
        out = subprocess.run('i-pi-remdsort input.xml', shell=True, capture_output=True, text=True)
        assert out.returncode == 0  # TODO: what if it isn't?
        for path in Path.cwd().glob('*.trajectory*.ase'):
            path.rename(path.with_suffix('.extxyz'))
        print('REMDSORT')

    # move recorded simulation observables
    files = args.output_props.split(",")
    assert len(states) == len(files)
    for idx, file in enumerate(files):
        file_src = next(Path.cwd().glob(f'{prefix}*{idx}*.properties'))
        file_src.rename(file)
    print('MOVED SIMULATION OBSERVABLES')

    if args.output_trajs is None:
        return

    # move trajectories
    files = args.output_trajs.split(",")
    assert len(states) == len(files)
    for idx, file in enumerate(files):
        file_src = next(Path.cwd().glob(f'{prefix}*{idx}*.trajectory*.extxyz'))
        atoms = read(file_src, ':')
        periodic = states[idx].periodic
        for at in atoms:
            if not periodic:  # load and replace cell
                at.pbc, at.cell = False, None
            at.info.pop("ipi_comment", None)
        write(file, atoms)
    print('MOVED SIMULATION TRAJECTORIES')

    return


def main():
    signal.signal(signal.SIGTERM, timeout_handler)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='what to do', dest='action')
    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "--nwalkers",
        type=int,
        default=None,
    )
    run_parser.add_argument(
        "--input_xml",
        type=str,
        default=None,
    )
    run_parser.add_argument(
        "--start_xyz",
        type=str,
        default=None,
    )
    clean_parser = subparsers.add_parser("cleanup")
    clean_parser.set_defaults(func=cleanup)
    clean_parser.add_argument(
        "--output_xyz",
        type=str,
        default=None,
    )
    clean_parser.add_argument(
        "--output_trajs",
        type=str,
        default=None,
    )
    clean_parser.add_argument(
        "--output_props",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.func(args)
    # try:
    #     args.func(args)
    # except Exception as e:  # noqa: B036
    #     print(e)
    #     print("i-Pi simulation failed!")
    #     print("files in directory:")
    #     for filepath in Path.cwd().glob("*"):
    #         print(filepath)

