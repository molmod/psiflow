class BaseModel:
    """Base class for a trainable interaction potential"""

    def get_calculator(self, device, dtype):
        raise NotImplementedError

    @staticmethod
    def train(model, training_execution, dataset):
        raise NotImplementedError

