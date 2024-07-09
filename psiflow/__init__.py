from pathlib import Path

import typeguard

from .config import setup_slurm_config  # noqa: F401
from .execution import ExecutionContextLoader
from .serialization import (  # noqa: F401
    _DataFuture,
    deserialize,
    serializable,
    serialize,
)


@typeguard.typechecked
def resolve_and_check(path: Path) -> Path:
    path = path.resolve()
    if Path.cwd() in path.parents:
        pass
    elif path.exists() and Path.cwd().samefile(path):
        pass
    else:
        raise ValueError(
            "requested file and/or path at location: {}"
            "\nwhich is not in the present working directory: {}"
            "\npsiflow can only load and/or save in its present "
            "working directory because this is the only directory"
            " that will get bound into the container.".format(path, Path.cwd())
        )
    return path


load = ExecutionContextLoader.load
context = ExecutionContextLoader.context
wait = ExecutionContextLoader.wait
