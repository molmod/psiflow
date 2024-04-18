import argparse
import ast
import glob
import shutil
import signal
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.geometry import Cell
from ase.io import read, write
from ase.units import Bohr, Ha, _e, _hbar, kB
from ipi.engine.outputs import CheckpointOutput, PropertyOutput, TrajectoryOutput
from ipi.engine.properties import getkey
from ipi.engine.simulation import Simulation
from ipi.inputs.simulation import InputSimulation
from ipi.utils.io.inputs import io_xml
from ipi.utils.messages import verbosity
from ipi.utils.softexit import softexit

from psiflow.data import Geometry
from psiflow.data.geometry import _write_frames

NONPERIODIC_CELL = 1000 * np.eye(3)


def remdsort(inputfile, prefix="SRT_"):
    verbosity.level = "low"
    # opens & parses the input file
    ifile = open(inputfile, "r")
    xmlrestart = io_xml.xml_parse_file(ifile)  # Parses the file.
    ifile.close()

    # ugly hack to remove ffplumed objects to avoid messing up with plumed output files
    newfields = [f for f in xmlrestart.fields[0][1].fields if f[0] != "ffplumed"]
    xmlrestart.fields[0][1].fields = newfields

    isimul = InputSimulation()
    isimul.parse(xmlrestart.fields[0][1])

    simul = isimul.fetch()
    swapfile = ""
    if simul.smotion is None or (
        simul.smotion.mode != "remd" and simul.smotion.mode != "multi"
    ):
        raise ValueError("Simulation does not look like a parallel tempering one.")
    else:
        if simul.smotion.mode == "remd":
            swapfile = simul.smotion.swapfile
        else:
            for sm in simul.smotion.mlist:
                if sm.mode == "remd":
                    if swapfile != "":
                        raise ValueError(
                            "I'm not equipped to deal with multiple REMD outputs, sorry"
                        )
                    swapfile = sm.swapfile
        if swapfile == "":
            raise ValueError(
                "Could not determine the REMD swapfile name. \
                 Sorry, you'll have to look carefully at your inputs."
            )

    # reconstructs the list of the property and trajectory files that have been output
    # and that should be re-ordered
    lprop = []  # list of property files
    ltraj = []  # list of trajectory files
    nsys = len(simul.syslist)
    for o in simul.outtemplate:
        o = deepcopy(o)  # avoids overwriting the actual filename
        if simul.outtemplate.prefix != "":
            o.filename = simul.outtemplate.prefix + "." + o.filename
        if (
            type(o) is CheckpointOutput
        ):  # properties and trajectories are output per system
            pass
        elif type(o) is PropertyOutput:
            nprop = []
            isys = 0
            for s in simul.syslist:  # create multiple copies
                if s.prefix != "":
                    filename = s.prefix + "_" + o.filename
                else:
                    filename = o.filename
                ofilename = prefix + filename
                nprop.append(
                    {
                        "filename": filename,
                        "ofilename": ofilename,
                        "stride": o.stride,
                        "ifile": open(filename, "r"),
                        "ofile": open(ofilename, "w"),
                    }
                )
                isys += 1
            lprop.append(nprop)
        elif (
            type(o) is TrajectoryOutput
        ):  # trajectories are more complex, as some have per-bead output
            if getkey(o.what) in [
                "positions",
                "velocities",
                "forces",
                "extras",
            ]:  # multiple beads
                nbeads = simul.syslist[0].beads.nbeads
                for b in range(nbeads):
                    ntraj = []
                    isys = 0
                    # zero-padded bead number
                    padb = (
                        "%0" + str(int(1 + np.floor(np.log(nbeads) / np.log(10)))) + "d"
                    ) % (b)
                    for s in simul.syslist:
                        if s.prefix != "":
                            filename = s.prefix + "_" + o.filename
                        else:
                            filename = o.filename
                        ofilename = prefix + filename
                        if o.ibead < 0 or o.ibead == b:
                            if getkey(o.what) == "extras":
                                filename = filename + "_" + padb
                                ofilename = ofilename + "_" + padb
                                # Sets format of extras as None
                                ntraj.append(
                                    {
                                        "filename": filename,
                                        "format": None,
                                        "ofilename": ofilename,
                                        "stride": o.stride,
                                        "ifile": open(filename, "r"),
                                        "ofile": open(ofilename, "w"),
                                    }
                                )
                            else:
                                filename = filename + "_" + padb + "." + o.format
                                ofilename = ofilename + "_" + padb + "." + o.format
                                ntraj.append(
                                    {
                                        "filename": filename,
                                        "format": o.format,
                                        "ofilename": ofilename,
                                        "stride": o.stride,
                                        "ifile": open(filename, "r"),
                                        "ofile": open(ofilename, "w"),
                                    }
                                )
                        isys += 1
                    if ntraj != []:
                        ltraj.append(ntraj)

            else:
                ntraj = []
                isys = 0
                for s in simul.syslist:  # create multiple copies
                    if s.prefix != "":
                        filename = s.prefix + "_" + o.filename
                    else:
                        filename = o.filename
                    filename = filename + "." + o.format
                    ofilename = prefix + filename
                    ntraj.append(
                        {
                            "filename": filename,
                            "format": o.format,
                            "ofilename": ofilename,
                            "stride": o.stride,
                            "ifile": open(filename, "r"),
                            "ofile": open(ofilename, "w"),
                        }
                    )

                    isys += 1
                ltraj.append(ntraj)

    ptfile = open(simul.outtemplate.prefix + "." + swapfile, "r")

    # now reads files one frame at a time,
    # and re-direct output to the appropriate location

    line = ptfile.readline().split()
    irep = list(range(nsys))  # Could this be harmful?
    step = 0
    while True:
        # reads one line from index file
        try:
            for prop in lprop:
                for isys in range(nsys):
                    sprop = prop[isys]
                    if step % sprop["stride"] == 0:  # property transfer
                        iline = sprop["ifile"].readline()
                        if len(iline) == 0:
                            raise EOFError  # useful if line is blank
                        while iline[0] == "#":  # fast forward if line is a comment
                            prop[irep[isys]]["ofile"].write(iline)
                            iline = sprop["ifile"].readline()
                        prop[irep[isys]]["ofile"].write(iline)
            for traj in ltraj:
                for isys in range(nsys):
                    straj = traj[isys]
                    if step % straj["stride"] == 0:  # property transfer
                        # reads one frame from the input file
                        ibuffer = []
                        if straj["format"] is None:
                            ibuffer.append(straj["ifile"].readline())
                            ibuffer.append(straj["ifile"].readline())
                            traj[irep[isys]]["ofile"].write("".join(ibuffer))
                        if straj["format"] in ["xyz", "ase"]:
                            iline = straj["ifile"].readline()
                            nat = int(iline)
                            ibuffer.append(iline)
                            ibuffer.append(straj["ifile"].readline())
                            for _i in range(nat):
                                ibuffer.append(straj["ifile"].readline())
                            traj[irep[isys]]["ofile"].write("".join(ibuffer))
                        elif straj["format"] == "pdb":
                            iline = straj["ifile"].readline()
                            while iline.strip() != "" and iline.strip() != "END":
                                ibuffer.append(iline)
                                iline = straj["ifile"].readline()
                            ibuffer.append(iline)
                            traj[irep[isys]]["ofile"].write("".join(ibuffer))
        except EOFError:
            break

        if len(line) > 0 and step == int(line[0]):
            irep = [int(i) for i in line[1:]]
            line = ptfile.readline()
            line = line.split()

        step += 1


def parse_checkpoint(
    checkpoint: ET.ElementTree,
) -> tuple[list[Geometry], np.ndarray]:
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
            numbers = [atomic_numbers[s] for s in symbols]

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
            try:
                time = float(list(ensemble.iter(tag="time"))[0].text) * conversion
            except IndexError:
                time = 0.0

            geometry = Geometry.from_data(
                numbers=np.array(numbers),
                positions=np.mean(positions, axis=0),
                cell=box,
            )
            geometry.order["temperature"] = temperature
            geometry.order["time"] = time
            states.append(geometry)
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
            address.text = str(Path.cwd() / address.text.strip())


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
        atoms = data_start[i]
        if not sum(atoms.pbc):  # set fake large cell for i-PI
            atoms.pbc = True
            atoms.cell = Cell(NONPERIODIC_CELL)
        write("start_{}.xyz".format(i), data_start[i])

    with open(args.input_xml, "r") as f:
        input_xml = ET.fromstring(f.read())

    insert_data_start(input_xml, data_start)
    insert_addresses(input_xml)
    with open("input.xml", "wb") as f:
        f.write(ET.tostring(input_xml, encoding="utf-8"))

    simulation = Simulation.load_from_xml("input.xml", sockets_prefix="")
    try:
        simulation.run()
        softexit.trigger(status="success", message=" @ SIMULATION: Exiting cleanly.")
    except TimeoutException:
        print("simulation timed out -- killing gracefully")


def cleanup(args):
    with open("input.xml", "r") as f:
        content = f.read()
    if "vibrations" in content:
        # do stuff
        pass
    else:
        checkpoint = ET.parse("output.checkpoint")
        states = parse_checkpoint(checkpoint)
        for state in states:
            if np.allclose(state.cell, NONPERIODIC_CELL):
                state.cell[:] = 0.0
        _write_frames(*states, outputs=[args.output_xyz])
        if "remd" in content:
            remdsort("input.xml")
            for filepath in glob.glob("SRT_*"):
                source = filepath
                target = filepath.replace("SRT_", "")
                assert Path(target).exists()  # should exist
                shutil.copyfile(source, target)
        i = 0
        while True:
            assert i <= len(states)
            path = Path("walker-{}_output.trajectory_0.ase".format(i))
            if path.exists() and not states[i].periodic:  # load and replace cell
                traj = read(path, index=":", format="xyz")
                path.unlink()
                for atoms in traj:
                    atoms.pbc = False
                    atoms.cell = None
                write(path, traj, format="xyz")
            else:
                break
            i += 1


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
