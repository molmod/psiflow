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
        ncores=4,
        dtype='float32',
        walltime=2,
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=1,
        walltime=3, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=4,
        omp_num_threads=1,
        mpi_command=lambda x: f'mpirun -np {x} -bind-to core',
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
    launcher = ContainerizedLauncher(uri='docker://svandenhaute/psiflow:1.0.1-rocm5.2', enable_gpu=False)
else:
    launcher = SimpleLauncher()

providers = {
        'default': LocalProvider(launcher=launcher),
        'model': LocalProvider(launcher=launcher, worker_init='export APPTAINERENV_OMP_PROC_BIND=CLOSE;'),
        'training': LocalProvider(launcher=launcher),
        'reference': LocalProvider(launcher=launcher),
        }


def get_config(path_internal):
    config = generate_parsl_config(
            path_internal,
            definitions,
            providers,
            use_work_queue=False,
            )
    return config, definitions
