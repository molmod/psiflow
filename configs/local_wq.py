from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher

from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference, HybridCP2KReference, \
        MP2CP2KReference, DoubleHybridCP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config, ContainerizedLauncher


model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=1,
        dtype='float32',
        walltime=2, # only applies to dynamic walkers
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=4,
        walltime=3, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=4,
        omp_num_threads=1,
        #mpi_command=lambda x: f'mpirun -np {x} --map-by ppr:{x}:node:PE=1:SPAN:NOOVERSUBSCRIBE',
        mpi_command=lambda x: f'mpirun -np {x} --map-by node:PE=1',
        cp2k_exec='cp2k.psmp',
        walltime=3, # in minutes
        )

definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        AllegroModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        HybridCP2KReference: [reference_evaluate],
        DoubleHybridCP2KReference: [reference_evaluate],
        MP2CP2KReference: [reference_evaluate],
        }

containerize = True
if containerize:
    launcher = ContainerizedLauncher(tag='1.0.0rc0-cuda11.3', enable_gpu=True)
else:
    launcher = SimpleLauncher()

providers = {
        'default': LocalProvider(launcher=launcher),
        'model': LocalProvider(launcher=launcher),
        'training': LocalProvider(launcher=launcher),
        'reference': LocalProvider(launcher=launcher),
        }


def get_config(path_parsl_internal):
    config = generate_parsl_config(
            path_parsl_internal,
            definitions,
            providers,
            use_work_queue=True,
            )
    return config, definitions
