import argparse
import traceback
import xml.etree.ElementTree as ET
import subprocess
import time
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
from psiflow.sampling.utils import create_xml_list

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




def wait_for_clients(input_xml, timeout: int = 60) -> None:
    """Make sure clients have initialised successfully"""

    # find sockets opened by server
    sockets = []
    xml_str = ET.tostring(input_xml, encoding="unicode")
    for line in xml_str.splitlines():
        if 'address' in line:
            sockets.append(line.split(">")[1].split("<")[0])

    for _ in range(timeout):
        # check socket status
        out = subprocess.check_output("ss -xpl | grep psiflow", shell=True).decode()
        connections = {}
        for line in out.strip().split("\n"):
            data = line.split()
            connections[data[4]] = (data[1], int(data[2]))

        # does every socket have a pending handshake
        pending = [connections.get(s, [None, 0])[1] for s in sockets]
        if all([i > 0 for i in pending]):
            return
        time.sleep(1)

    msg = f"Timed out waiting for clients to initialise. Sockets: {connections}"
    raise ConnectionError(msg)


def run(start_xyz: str, input_xml: str):
    # prepare starting geometries from context_dir
    data_start: list[ase.Atoms] = read(start_xyz, index=":")
    for i, at in enumerate(data_start):
        print(at.pbc)
        if not any(at.pbc):  # set fake large cell for i-PI
            at.pbc = True
            at.cell = Cell(NONPERIODIC_CELL)
        write("start_{}.xyz".format(i), data_start[i])

    with open(input_xml, "r") as f:
        input_xml = ET.fromstring(f.read())

    insert_addresses(input_xml)
    with open(INPUT_XML, "wb") as f:
        f.write(ET.tostring(input_xml, encoding="utf-8"))

    simulation = Simulation.load_from_xml(open(INPUT_XML), sockets_prefix="")
    wait_for_clients(input_xml)
    simulation.run()
    return


def cleanup(output_xyz: str, output_props: str, output_trajs: str) -> None:
    from psiflow.data.utils import _write_frames

    print("Starting cleanup")
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
    print("Moved checkpoint geometries")

    prefix = ""
    if "remd" in content:
        # unshuffle simulation output according to ensemble
        prefix = "SRT_"
        out = subprocess.run(
            "i-pi-remdsort input.xml", shell=True, capture_output=True, text=True
        )
        assert out.returncode == 0  # TODO: what if it isn't?
        print("REMDSORT")

    output_props = _.split(",") if (_ := output_props) else []
    output_trajs = _.split(",") if (_ := output_trajs) else []

    # move recorded simulation observables
    if len(output_props):
        assert len(states) == len(output_props)
        for idx, file in enumerate(output_props):
            file_src = next(Path.cwd().glob(f"{prefix}*{idx}*.properties"))
            Path(file).write_text(file_src.read_text())
        print("Moved simulation observables")

    if len(output_trajs) == 0:
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
    print("Moved simulation trajectories")

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xml", required=True)
    parser.add_argument("--start_xyz", required=True)
    parser.add_argument("--output_xyz", required=True)
    parser.add_argument("--output_props", default="")
    parser.add_argument("--output_trajs", default="")
    args = parser.parse_args()

    time_start = time.time()
    try:
        run(args.start_xyz, args.input_xml)
        softexit.trigger(status="success", message="@PSIFLOW: We are done here.")
    except ConnectionError:
        # TODO: in this case, no output files are generated..
        traceback.print_exc()
        softexit.trigger(status="bad", message="@PSIFLOW: Clients failed to connect.")
    except np.linalg.LinAlgError:
        # some NaN / INF value appeared
        traceback.print_exc()
        softexit.trigger(status="bad", message="@PSIFLOW: Simulation went boom.")
    except SystemExit:
        # i-Pi intercepts SIG_INT and SIG_TERM by default,
        # merges all threads and calls sys.exit() before we can clean up
        pass
    finally:
        cleanup(args.output_xyz, args.output_props, args.output_trajs)
        time_stop = time.time()
        print("|- Done -|")
        print(f"Total simulation time: {time_stop - time_start:.0f} seconds")
