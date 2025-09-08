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
from ipi.engine.simulation import Simulation
from ipi.utils.softexit import softexit

from psiflow.geometry import Geometry
from psiflow.sampling.utils import TimeoutException, timeout_handler, create_xml_list

INPUT_XML = "input.xml"
NONPERIODIC_CELL = 1000 * np.eye(3)


def parse_checkpoint(file_xml: str | Path) -> list[Geometry]:
    """"""
    from ipi.inputs.system import InputSystem
    from ipi.utils.io.inputs.io_xml import xml_parse_file

    # see Simulation.load_from_xml
    xml_sim = xml_parse_file(open(file_xml)).fields[0][1]
    xml_systems = {v.attribs["prefix"]: v for k, v in xml_sim.fields if k == "system"}

    geometries = []
    for name, xml_sys in xml_systems.items():
        sys = InputSystem()
        sys.parse(xml_sys)

        # get structural properties
        beads = sys.beads.fetch()
        natoms, nbeads = beads.natoms, beads.nbeads  # average over beads
        positions = np.asarray(beads.q).reshape(nbeads, natoms, 3).mean(axis=0)
        numbers = np.array([atomic_numbers[s] for s in beads.names])
        cell = np.asarray(sys.cell.fetch().h).T  # transpose to undo i-Pi convention
        geometry = Geometry.from_data(numbers, positions * Bohr, cell * Bohr)

        # get current internal system time
        ensemble_data = sys.ensemble.fetch()
        time = ensemble_data.time * (_hbar / (_e * Ha)) * 1e12
        geometry.order["time"], geometry.order["name"] = time, name
        geometries.append(geometry)

    return geometries


def insert_addresses(input_xml: ET.Element) -> None:
    """Prepend working directory to socket names"""
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
        # TODO: what if cells are different?
        cell = np.array(data_start[0].cell).flatten(order="F")
        h0.text = create_xml_list([str(a) for a in cell])

        path = "system_template/template/system/ensemble"
        ensemble = input_xml.find(path)
        assert ensemble is not None
        pressure = ensemble.find("pressure")
        if pressure is not None:
            ensemble.remove(pressure)
            stress = ET.SubElement(ensemble, "stress", units="megapascal")
            stress.text = " [ PRESSURE, 0, 0, 0, PRESSURE, 0, 0, 0, PRESSURE ] "  # TODO: anisotropic?


def run(start_xyz: str, input_xml: str):
    # prepare starting geometries from context_dir
    data_start = read(start_xyz, index=":")
    for i, at in enumerate(data_start):
        if not any(at.pbc):  # set fake large cell for i-PI
            at.pbc = True
            at.cell = Cell(NONPERIODIC_CELL)
        write("start_{}.xyz".format(i), data_start[i])

    with open(input_xml, "r") as f:
        input_xml = ET.fromstring(f.read())

    insert_data_start(input_xml)
    insert_addresses(input_xml)
    anisotropic_barostat_h0(input_xml, data_start)
    with open(INPUT_XML, "wb") as f:
        f.write(ET.tostring(input_xml, encoding="utf-8"))

    simulation = Simulation.load_from_xml(open(INPUT_XML), sockets_prefix="")
    simulation.run()


def cleanup(
    output_xyz: str, output_props: list[str], output_trajs: list[str] | None = None
) -> None:
    from psiflow.data.utils import _write_frames

    print("STARTING CLEANUP")
    with open(INPUT_XML, "r") as f:
        content = f.read()
    if "vibrations" in content:
        # do stuff
        return

    # collect final checkpoint geometries
    states = parse_checkpoint("output.checkpoint")
    for state in states:
        if np.allclose(state.cell, NONPERIODIC_CELL):
            state.cell[:] = 0.0
    _write_frames(*states, outputs=[output_xyz])
    print("MOVED CHECKPOINT GEOMETRIES")

    prefix = ""
    if "remd" in content:
        # unshuffle simulation output according to ensemble
        prefix = "SRT_"
        out = subprocess.run(
            "i-pi-remdsort input.xml", shell=True, capture_output=True, text=True
        )
        assert out.returncode == 0  # TODO: what if it isn't?
        print("REMDSORT")

    # move recorded simulation observables
    assert len(states) == len(output_props)
    for idx, file in enumerate(output_props):
        file_src = next(Path.cwd().glob(f"{prefix}*{idx}*.properties"))
        Path(file).write_text(file_src.read_text())
    print("MOVED SIMULATION OBSERVABLES")

    if output_trajs is None:
        return

    # move trajectories
    assert len(states) == len(output_trajs)
    for idx, file in enumerate(output_trajs):
        file_src = next(Path.cwd().glob(f"{prefix}*{idx}*.trajectory*.extxyz"))
        atoms = read(file_src, ":")
        periodic = states[idx].periodic
        for at in atoms:
            if not periodic:  # load and replace cell
                at.pbc, at.cell = False, None
            at.info.pop("ipi_comment", None)
        write(file, atoms)
    print("MOVED SIMULATION TRAJECTORIES")

    return


def main():
    signal.signal(signal.SIGTERM, timeout_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xml", required=True)
    parser.add_argument("--start_xyz", required=True)
    parser.add_argument("--output_xyz", required=True)
    parser.add_argument("--output_props", required=True)
    parser.add_argument("--output_trajs", default="")
    args = parser.parse_args()

    try:
        run(args.start_xyz, args.input_xml)
        softexit.trigger(status="success", message="@PSIFLOW: We are done here.")
    except TimeoutException:
        softexit.trigger(message="@PSIFLOW: Timeout. Saving intermediate progress.")
    except SystemExit:
        # i-Pi merges all threads and calls sys.exit() before we can clean up
        output_props = args.output_props.split(",")
        output_trajs = args.output_trajs.split(",")
        cleanup(args.output_xyz, output_props, output_trajs)
