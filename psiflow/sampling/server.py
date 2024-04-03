import argparse
import ast
import signal
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.units import Bohr, Ha, _e, _hbar, kB
from ipi.engine.simulation import Simulation
from ipi.utils.softexit import softexit

from psiflow.data import FlowAtoms


def parse_checkpoint(
    checkpoint: ET.ElementTree,
) -> tuple[list[FlowAtoms], np.ndarray]:
    systems = {s.attrib["prefix"]: s for s in checkpoint.iter(tag="system")}

    states = []
    walker_index = 0
    while True:
        system = systems.get("walker-{}".format(walker_index), None)
        if system is not None:
            beads = list(system.iter(tag="beads"))[0]
            natoms = int(beads.attrib["natoms"])
            nbeads = int(beads.attrib["nbeads"])

            text = "".join(list(beads.iter(tag="q"))[0].text.split())
            positions = np.array(ast.literal_eval(text))
            positions = positions.reshape(nbeads, natoms, 3) * Bohr

            text = "".join(list(beads.iter(tag="names"))[0].text.split())
            text = text.replace(
                ",",
                '","',
            )
            text = text.replace(
                "[",
                '["',
            )
            text = text.replace(
                "]",
                '"]',
            )
            symbols = ast.literal_eval(text)

            text = "".join(list(system.iter(tag="cell"))[0].text.split())
            box = (
                np.array(ast.literal_eval(text)).reshape(3, 3).T * Bohr
            )  # transpose for convention

            # parse temperature
            text = "".join(list(beads.iter(tag="p"))[0].text.split())
            momenta = np.array(ast.literal_eval(text))
            momenta = momenta.reshape(nbeads, natoms, 3)
            text = "".join(list(beads.iter(tag="m"))[0].text.split())
            masses = np.array(ast.literal_eval(text)).reshape(1, natoms, 1)

            kinetic_per_dof = np.mean(momenta**2 / (2 * masses))
            kinetic_per_dof *= Ha
            temperature = 2 * kinetic_per_dof / kB

            # get current internal system time
            ensemble = list(system.iter(tag="ensemble"))[0]
            conversion = (_hbar / (_e * Ha)) * 1e12
            time = float(list(ensemble.iter(tag="time"))[0].text) * conversion

            atoms = FlowAtoms(
                symbols=symbols,
                positions=np.mean(positions, axis=0),
                pbc=True,
                cell=box,
            )
            atoms.info["temperature"] = temperature
            atoms.info["time"] = time
            states.append(atoms)
            walker_index += 1
        else:
            break
    return states


def insert_addresses(input_xml):
    for child in input_xml:
        if child.tag == "ffsocket":
            for child_ in child:
                if child_.tag == "address":
                    address = child_
                    break
            address.text = Path.cwd().name[4:] + "/" + address.text.strip()


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def insert_data_start(input_xml, data_start):
    for child in input_xml:
        if child.tag == "system":
            initialize = ET.Element("initialize", nbeads="1")
            initialize.text = "start_INDEX.xyz"


def start(args):
    data_start = read(args.start_xyz, index=":")
    assert len(data_start) == args.nwalkers
    for i in range(args.nwalkers):
        write("start_{}.xyz".format(i), data_start[i])

    with open(args.input_xml, "r") as f:
        input_xml = ET.fromstring(f.read())

    insert_data_start(input_xml, data_start)
    insert_addresses(input_xml)
    with open("input.xml", "wb") as f:
        f.write(ET.tostring(input_xml, encoding="utf-8"))

    simulation = Simulation.load_from_xml("input.xml")
    try:
        simulation.run()
        softexit.trigger(status="success", message=" @ SIMULATION: Exiting cleanly.")
    except TimeoutException:
        print("simulation timed out -- killing gracefully")


def cleanup(args):
    checkpoint = ET.parse("output.checkpoint")
    states = parse_checkpoint(checkpoint)
    write(args.output_xyz, states)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, timeout_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input_xml",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_xyz",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_xyz",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if not args.cleanup:
        start(args)
    else:
        cleanup(args)
