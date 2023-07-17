from parsl.executors import ThreadPoolExecutor
from parsl.config import Config

from psiflow.models import MACEModel, NequIPModel, AllegroModel
from psiflow.reference import CP2KReference, HybridCP2KReference, \
        MP2CP2KReference, DoubleHybridCP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution


model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=1,
        dtype='float32',
        walltime=1,
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=1,
        walltime=1, # in minutes; includes 100s slack
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=4,
        omp_num_threads=2,
        mpi_command=lambda x: f'mpirun -np {x}',
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

def get_config(path_internal):
    executors = [
            ThreadPoolExecutor(label='default', max_threads=1, working_dir=str(path_internal)),
            ThreadPoolExecutor(label=model_training.executor, max_threads=model_training.ncores, working_dir=str(path_internal)),
            ThreadPoolExecutor(label=model_evaluate.executor, max_threads=model_evaluate.ncores, working_dir=str(path_internal)),
            ThreadPoolExecutor(label=reference_evaluate.executor, max_threads=reference_evaluate.ncores, working_dir=str(path_internal)),
            ]
    config = Config(executors, run_dir=str(path_internal), usage_tracking=True, app_cache=False)
    return config, definitions
