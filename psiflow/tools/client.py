import os
import argparse
from pathlib import Path

from psiflow.hamiltonians import deserialize_calculator
from psiflow.hamiltonians.utils import ForceMagnitudeException

# from ipi._driver.driver import run_driver
# from ipi._driver.pes.ase import ASEDriver


if __name__ == "__main__":
    print('OS environment values:')
    for key, value in os.environ.items():
        print(key, value)
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

    from ipi._driver.driver import run_driver
    from ipi._driver.pes.ase import ASEDriver

    # assert not args.start, args.start
    driver = ASEDriver(args=args.start)
    driver.ase_calculator = deserialize_calculator(
        args.path_hamiltonian,
        device=args.device,
        dtype=args.dtype,
        max_force=args.max_force,
    )
    import torch
    import psutil
    print('torch num threads: ', torch.get_num_threads())
    print('cpu count: ', psutil.cpu_count(logical=False))

    try:
        run_driver(
            unix=True,
            address=str(Path.cwd() / args.address),
            driver=driver,
            sockets_prefix="",
        )
    except ForceMagnitudeException as e:
        print(e)  # induce timeout in server
    except ConnectionResetError as e:  # some other client induced a timeout
        print(e)
