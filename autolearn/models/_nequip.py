from typing import Union, Optional, Callable, Dict, Final
from pathlib import Path
import tempfile
import yaml
import covalent as ct
import torch

from nequip.scripts.train import fresh_start, restart, default_config
from nequip.scripts.deploy import CONFIG_KEY, NEQUIP_VERSION_KEY, \
        TORCH_VERSION_KEY, E3NN_VERSION_KEY, R_MAX_KEY, N_SPECIES_KEY, \
        TYPE_NAMES_KEY, JIT_BAILOUT_KEY, JIT_FUSION_STRATEGY, TF32_KEY
from nequip.data import ASEDataset
from nequip.model import model_from_config
from nequip.utils import Config, instantiate
from nequip.utils._global_options import _set_global_options
from nequip.utils.versions import check_code_version, get_config_code_versions
from nequip.data.transforms import TypeMapper
from nequip.train.trainer import Trainer
from nequip.ase import NequIPCalculator
from e3nn.util.jit import script

from autolearn import Dataset
from .model import BaseModel
from autolearn.utils import prepare_dict


def to_nequip_dataset(data, config):
    _config = Config.from_dict(dict(config))
    _config['chemical_symbols'] = Dataset(data).get_elements()
    type_mapper, _ = instantiate(
            TypeMapper,
            prefix='dataset',
            optional_args=_config,
            )
    ase_dataset = ASEDataset.from_atoms_list(
            data,
            extra_fixed_fields={'r_max': config['r_max']},
            type_mapper=type_mapper,
            )
    return ase_dataset


def get_train_electron(training_execution):
    device = training_execution.device
    def train_barebones(nequip_model, dataset):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            nequip_config_dict = nequip_model.nequip_config
            nequip_config_dict['device'] = device
            nequip_config_dict['root'] = tmpdir
            nequip_config = Config.from_dict(nequip_config_dict)
            model = nequip_model.model.to(device)

            check_code_version(nequip_config, add_to_config=True)
            _set_global_options(nequip_config)
            trainer = Trainer(model=None, **dict(nequip_config))
            trainer.n_train = len(dataset.training)
            trainer.n_val   = len(dataset.validation)

            data_train    = to_nequip_dataset(dataset.training, nequip_config)
            data_validate = to_nequip_dataset(dataset.validation, nequip_config)
            trainer.set_dataset(data_train, data_validate)
            trainer.model = model

            # Store any updated config information in the trainer
            trainer.update_kwargs(nequip_config)
            trainer.train()
        nequip_model.model = trainer.model.to('cpu')
        return nequip_model
    return ct.electron(train_barebones, executor=training_execution.executor)


class NequIPModel(BaseModel):
    """Model wrapper for NequIP"""

    def __init__(self, config):
        """Constructor"""
        self.config = dict(config)
        self.config['device'] = 'cpu' # force CPU

        # only initialized based on dataset
        self.nequip_config = None
        self.model         = None

    def initialize(self, dataset, path_model=None):
        nequip_config = Config.from_dict(
                self.config,
                defaults=default_config,
                )
        ase_dataset = to_nequip_dataset(dataset.training, nequip_config)
        self.model = model_from_config(
                nequip_config,
                initialize=True,
                dataset=ase_dataset,
                )
        self.nequip_config = nequip_config.as_dict()
        if path_model is not None:
            model_state_dict = torch.load(path_model, map_location='cpu')
            self.model.load_state_dict(model_state_dict)

    def get_calculator(self, device, dtype):
        model = self.model
        if dtype == 'float64':
            model = model.double()

        # compile for deploy
        model.eval()
        if not isinstance(model, torch.jit.ScriptModule):
            model = script(model)

        # generate metadata
        nequip_config = Config.from_dict(self.nequip_config)
        metadata: dict = {}
        code_versions, code_commits = get_config_code_versions(nequip_config)
        for code, version in code_versions.items():
            metadata[code + "_version"] = version
        if len(code_commits) > 0:
            metadata[CODE_COMMITS_KEY] = ";".join(
                f"{k}={v}" for k, v in code_commits.items()
            )

        metadata[R_MAX_KEY] = str(float(nequip_config["r_max"]))
        if "allowed_species" in nequip_config:
            # This is from before the atomic number updates
            n_species = len(nequip_config["allowed_species"])
            type_names = {
                type: ase.data.chemical_symbols[atomic_num]
                for type, atomic_num in enumerate(nequip_config["allowed_species"])
            }
        else:
            # The new atomic number setup
            n_species = str(nequip_config["num_types"])
            type_names = nequip_config["type_names"]
        metadata[N_SPECIES_KEY] = str(n_species)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)

        metadata[JIT_BAILOUT_KEY] = str(nequip_config[JIT_BAILOUT_KEY])
        if int(torch.__version__.split(".")[1]) >= 11 and JIT_FUSION_STRATEGY in nequip_config:
            metadata[JIT_FUSION_STRATEGY] = ";".join(
                "%s,%i" % e for e in nequip_config[JIT_FUSION_STRATEGY]
            )
        metadata[TF32_KEY] = str(int(nequip_config["allow_tf32"]))
        metadata[CONFIG_KEY] = yaml.dump(dict(nequip_config))

        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            path_model = Path(tmpdir) / 'model.pth'
            torch.jit.save(model, path_model, _extra_files=metadata)
            calculator = NequIPCalculator.from_deployed_model(
                    path_model,
                    device=device,
                    )
        return calculator

    @staticmethod
    def train(nequip_model, training_execution, dataset):
        if nequip_model.model is None:
            print('initializing model ... ')
            nequip_model.initialize(dataset)

        train_electron = get_train_electron(training_execution)
        return train_electron(nequip_model, dataset)
