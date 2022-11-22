

class TrainingExecution:

    def __init__(self, executor='local', device='cuda'):
        self.executor = executor
        self.device   = device


class ModelExecution:

    def __init__(
            self,
            executor='local',
            device='cpu',
            ncores=1,
            dtype='float32',
            ):
        self.executor = executor
        self.device   = device
        self.ncores   = ncores
        self.dtype    = dtype


class EvaluatorExecution:

    def __init__(self, executor='local', ncores=1):
        self.executor = executor
        self.device   = 'cpu'
        self.ncores   = ncores
