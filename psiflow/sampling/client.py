import argparse
import socket

from ase.calculators.socketio import IPIProtocol, SocketClient
from ase.io import read

from psiflow.hamiltonians import deserialize_calculator
from psiflow.hamiltonians.utils import ForceMagnitudeException


class SimpleSocketClient(SocketClient):
    def __init__(
        self,
        host="localhost",
        port=None,
        unixsocket=None,
        timeout=None,
        log=None,
        comm=None,
    ):
        if comm is None:
            from ase.parallel import world

            comm = world

        self.comm = comm

        if self.comm.rank == 0:
            if unixsocket is not None:
                sock = socket.socket(socket.AF_UNIX)
                sock.connect(unixsocket)
            else:
                raise NotImplementedError
            sock.settimeout(timeout)
            self.host = host
            self.port = port
            self.unixsocket = unixsocket

            self.protocol = IPIProtocol(sock, txt=log)
            self.log = self.protocol.log
            self.closed = False

            self.bead_index = 0
            self.bead_initbytes = b""
            self.state = "READY"


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
    atoms.calc = deserialize_calculator(
        args.path_hamiltonian,
        device=args.device,
        dtype=args.dtype,
        max_force=args.max_force,
    )

    # address = Path.cwd().name[4:] + "/" + args.address.strip()
    client = SimpleSocketClient(unixsocket=args.address)
    try:
        client.run(atoms)
    except ForceMagnitudeException as e:
        print(e)
