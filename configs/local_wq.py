from parsl.providers import LocalProvider

from psiflow.models import MACEModel, NequIPModel
from psiflow.reference import CP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config


model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=1,
        dtype='float32',
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=2,
        walltime=3, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=4,
        omp_num_threads=2,
        mpi_command=lambda x: f'mpirun -np {x}',
        cp2k_exec='cp2k.psmp',
        walltime=1, # in minutes
        )

definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        }
providers = {
        'default': LocalProvider(),
        'model': LocalProvider(),
        'training': LocalProvider(),
        'reference': LocalProvider(),
        }


def get_config(path_parsl_internal):
    config = generate_parsl_config(
            path_parsl_internal,
            definitions,
            providers,
            use_work_queue=True,
            )
    return config, definitions
