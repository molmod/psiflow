import argparse
import time
from pathlib import Path

from ase.calculators.socketio import SocketClient
from ase.io import read

from psiflow.hamiltonians import deserialize
from psiflow.hamiltonians.utils import ForceMagnitudeException

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_hamiltonian",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_force",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    assert args.path_hamiltonian is not None
    assert args.address is not None
    assert args.start is not None

    atoms = read(args.start)
    atoms.calc = deserialize(
        args.path_hamiltonian,
        device=args.device,
        dtype=args.dtype,
        max_force=None,
    )

    address = Path.cwd().name[4:] + "/" + args.address.strip()
    client = SocketClient(unixsocket=address)
    try:
        client.run(atoms)
    except ForceMagnitudeException as e:
        print(e)
        time.sleep(60)  # trigger i-PI timeout
