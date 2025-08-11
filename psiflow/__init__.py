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


# TODO: EXECUTION
#  - max_runtime is in seconds, max_simulation_time (and others) is in minutes
#  - ExecutionDefinition gpu argument?
#  - ExecutionDefinition wrap_in_timeout functionality
#  - ExecutionDefinition wrap_in_srun functionality? Actually for MD this is more iffy right
#  - ExecutionDefinition why properties?
#  - ExecutionDefinition centralize wq_resources
#  - configuration file with all options
#  - timeout -s 9 or -s 15?
#  - executor keys are hardcoded strings..
#  - define a 'format_env_variables' util
#  - update GPAW + ORCA + Default containers
#    include s-dftd3 in modelevaluation + install s-dftd3 with openmp
#  - what with mem_per_node / mem_per_worker
#  - always GPU for training?
#  - currently reference mpi_args have to be tuple according to typeguard..
#  - cores_per_block has to be specified even when exclusive..?
#  - can we do something with WQ priority?
#  - see chatgpt convo for process memory limits and such
#  - make /tmp for app workdirs an option?
#  - what is scaling_cores_per_worker in WQ
#  -
# TODO: REFERENCE
#  - reference MPI args not really checked
#  - mpi flags are very finicky across implementations --> use ENV args?
#    OMPI_MCA_orte_report_bindings=1 I_MPI_DEBUG=4
#  - commands ends with 'exit 0' - what if we do not want to exit yet?
#  - some actual logging?
#  - safe_compute_dataset functionality?
#  -
# TODO: MISC
#  - think about test efficiency
#  - some imports take a very long time
#  - fix serialisation
#  -
