from ase.data import chemical_symbols


def get_evaluate_electron(model_execution):
    device = model_execution.device
    ncores = model_execution.ncores
    dtype  = model_execution.dtype
    def evaluate_barebones(model, dataset):
        if device == 'cpu':
            torch.set_num_threads(ncores)
        _all = dataset.training + dataset.validation
        assert len(dataset.training) + len(dataset.validation) > 0
        atoms = _all[0].copy()
        atoms.calc = model.get_calculator(device, dtype)
        for state in _all:
            state.calc = None
            atoms.set_positions(state.get_positions())
            atoms.set_cell(state.get_cell())
            state.info['energy']   = atoms.get_potential_energy()
            state.info['stress']   = atoms.get_stress(voigt=False)
            state.arrays['forces'] = atoms.get_forces()
        return dataset
    return ct.electron(evaluate_barebones, executor=model_execution.executor)


class Dataset:
    """Dataset"""

    def __init__(self, training, validation=[]):
        self.training   = training
        self.validation = validation

    def add(self, dataset):
        self.training   += dataset.training
        self.validation += dataset.validation

    def get_numbers(self):
        """Returns set of all atomic numbers that are present in the data"""
        _all = [set(a.numbers) for a in (self.training + self.validation)]
        return sorted(list(set(b for a in _all for b in a)))

    def get_elements(self):
        numbers = self.get_numbers()
        return [chemical_symbols[n] for n in numbers]

    @staticmethod
    def evaluate(model, model_execution, dataset):
        evaluate_electron = get_evaluate_electron(model_execution)
        return evaluate_electron(model, dataset)


class HomogeneousDataset(Dataset):

    def __init__(self, training, validation=[]):
        super().__init__(training, validation)
        self.ntrain = len(training)
        self.nvalidate = len(validation)

        n = self.ntrain + self.nvalidate
        self.energy = np.zeros(n)
        self.forces = np.zeros((n, len(training[0]), 3))
        self.stress = np.zeros((n, 3, 3))

    def update_arrays(self):
        _all = self.training + self.validation
        for i, atoms in enumerate(_all):
            self.energy[i]    = _all.info['energy']
            self.stress[i, :] = _all.info['stress']
            self.forces[i, :] = _all.arrays['forces']

    def add(self, dataset):
        super().add(dataset)
        self.update_arrays()

    @staticmethod
    def compute_mae(data_0, data_1):
        pass
