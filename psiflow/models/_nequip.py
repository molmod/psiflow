from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List, Any, Dict
import typeguard
import logging
import inspect
from pathlib import Path

from ase.calculators.calculator import BaseCalculator

import parsl
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo

from psiflow.models.base import evaluate_dataset
from psiflow.models import BaseModel
from psiflow.data import FlowAtoms, Dataset
from psiflow.execution import ExecutionContext
from psiflow.utils import copy_data_future


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


def init_n_update(config, tmpdir):
    import wandb
    import logging
    from wandb.util import json_friendly_val
    conf_dict = dict(config)
    # wandb mangles keys (in terms of type) as well, but we can't easily correct that because there are many ambiguous edge cases. (E.g. string "-1" vs int -1 as keys, are they different config keys?)
    if any(not isinstance(k, str) for k in conf_dict.keys()):
        raise TypeError(
            "Due to wandb limitations, only string keys are supported in configurations."
        )

    # download from wandb set up
    config.run_id = wandb.util.generate_id()

    wandb.init(
        project=config.wandb_project,
        config=conf_dict,
        name=config.run_name,
        dir=tmpdir,
        resume="allow",
        id=config.run_id,
    )
    # # download from wandb set up
    updated_parameters = dict(wandb.config)
    for k, v_new in updated_parameters.items():
        skip = False
        if k in config.keys():
            # double check the one sanitized by wandb
            v_old = json_friendly_val(config[k])
            if repr(v_new) == repr(v_old):
                skip = True
        if skip:
            logging.info(f"# skipping wandb update {k} from {v_old} to {v_new}")
        else:
            config.update({k: v_new})
            logging.info(f"# wandb update {k} from {v_old} to {v_new}")
    return config


@typeguard.typechecked
def get_elements(data: List[FlowAtoms]) -> List[str]:
    from ase.data import chemical_symbols
    _all = [set(a.numbers) for a in data]
    numbers = sorted(list(set(b for a in _all for b in a)))
    return [chemical_symbols[n] for n in numbers]

# do not type hint ASEDataset to avoid having to import nequip types outside
# of the function
@typeguard.typechecked
def to_nequip_dataset(data: List[FlowAtoms], nequip_config: Any):
    import tempfile
    import shutil
    from copy import deepcopy
    from nequip.utils import Config, instantiate
    from nequip.data.transforms import TypeMapper
    from nequip.data import ASEDataset

    tmpdir = tempfile.mkdtemp() # not guaranteed to be new/empty for some reason
    shutil.rmtree(tmpdir)
    Path(tmpdir).mkdir()
    nequip_config_dict = deepcopy(dict(nequip_config))
    nequip_config_dict['root'] = tmpdir
    _config = Config.from_dict(dict(nequip_config_dict))
    _config['chemical_symbols'] = get_elements(data)
    type_mapper, _ = instantiate(
            TypeMapper,
            prefix='dataset',
            optional_args=_config,
            )
    ase_dataset = ASEDataset.from_atoms_list(
            data,
            extra_fixed_fields={'r_max': _config['r_max']},
            type_mapper=type_mapper,
            include_keys=_config['dataset_include_keys'],
            key_mapping=_config['dataset_key_mapping'],
            )
    shutil.rmtree(tmpdir)
    return ase_dataset


@typeguard.typechecked
def initialize(
        config: Dict,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> Dict:
    import torch
    import numpy as np
    from nequip.utils import Config
    from nequip.scripts.train import default_config
    from nequip.model import model_from_config

    from psiflow.data import read_dataset
    from psiflow.models._nequip import to_nequip_dataset

    torch.manual_seed(config['seed']) # necessary to ensure reproducible init!
    np.random.seed(config['seed'])

    nequip_config = Config.from_dict(
            config,
            defaults=default_config,
            )
    ase_dataset = to_nequip_dataset(
            read_dataset(slice(None), inputs=[inputs[0]]),
            nequip_config,
            )
    model = model_from_config(
            nequip_config,
            initialize=True,
            dataset=ase_dataset,
            )
    nequip_config = nequip_config.as_dict()
    torch.save(model.state_dict(), outputs[0].filepath)
    return nequip_config


@typeguard.typechecked
def deploy(
        device: str,
        dtype: str,
        nequip_config: Dict,
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> None:
    import torch
    import ase
    import yaml
    from nequip.utils import Config
    from nequip.model import model_from_config
    from e3nn.util.jit import script
    from nequip.scripts.deploy import CONFIG_KEY, NEQUIP_VERSION_KEY, \
            TORCH_VERSION_KEY, E3NN_VERSION_KEY, R_MAX_KEY, N_SPECIES_KEY, \
            TYPE_NAMES_KEY, JIT_BAILOUT_KEY, JIT_FUSION_STRATEGY, TF32_KEY
    from nequip.utils.versions import get_config_code_versions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # load model
    nequip_config = Config.from_dict(nequip_config)
    model = model_from_config(
            nequip_config,
            initialize=False,
            )
    model.load_state_dict(torch.load(inputs[0].filepath, map_location='cpu'))
    torch_device = torch.device(device)
    if dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64
    model.to(device=torch_device, dtype=torch_dtype)

    # compile for deploy
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    # generate metadata
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
    torch.jit.save(model, outputs[0].filepath, _extra_files=metadata)


@typeguard.typechecked
def train(
        device: str,
        dtype: str,
        nequip_config: Dict,
        inputs: List[File] = [],
        outputs: List[File] = [],
        walltime: float = 3600,
        ) -> int:
    import torch
    import tempfile
    import shutil
    from pathlib import Path
    from copy import deepcopy
    from nequip.utils import Config
    from nequip.model import model_from_config
    from nequip.utils.versions import check_code_version
    from nequip.utils._global_options import _set_global_options
    from nequip.train.trainer import Trainer
    from nequip.train.trainer_wandb import TrainerWandB
    from psiflow.data import read_dataset
    from psiflow.models._nequip import to_nequip_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nequip_config = Config.from_dict(nequip_config)
    tmpdir = tempfile.mkdtemp() # not guaranteed to be new/empty for some reason
    shutil.rmtree(tmpdir)
    Path(tmpdir).mkdir()
    nequip_config['root'] = tmpdir
    model = model_from_config(
            nequip_config,
            initialize=False,
            )
    model.load_state_dict(torch.load(inputs[0].filepath, map_location='cpu'))
    torch_device = torch.device(device)
    if dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64
    model.to(device=torch_device, dtype=torch_dtype)

    check_code_version(nequip_config, add_to_config=True)
    _set_global_options(nequip_config)
    if nequip_config.wandb:
        nequip_config = init_n_update(nequip_config, tmpdir)
        trainer_cls = TrainerWandB
    else:
        trainer_cls = Trainer
    assert dict(nequip_config)['root'] == tmpdir
    trainer = trainer_cls(model=None, **dict(nequip_config))
    training   = read_dataset(slice(None), inputs=[inputs[1]])
    validation = read_dataset(slice(None), inputs=[inputs[2]])
    trainer.n_train = len(training)
    trainer.n_val   = len(validation)

    data_train    = to_nequip_dataset(training, nequip_config)
    data_validate = to_nequip_dataset(validation, nequip_config)
    trainer.set_dataset(data_train, data_validate)
    trainer.model = model

    # Store any updated config information in the trainer
    trainer.update_kwargs(nequip_config)
    try:
        trainer.train()
    except parsl.app.errors.AppTimeout as e:
        # uncaught app timeouts are still possible before train.train(),
        # especially when large datasets need to be processed. 
        pass
    torch.save(trainer.model.to('cpu').state_dict(), outputs[0].filepath)
    return trainer.iepoch


@typeguard.typechecked
class NequIPModel(BaseModel):
    """Container class for NequIP models"""

    def __init__(self, context: ExecutionContext, config: Dict) -> None:
        config = dict(config)
        config['dataset_include_keys'] = ['total_energy', 'forces', 'virial']
        config['dataset_key_mapping'] = {
                'energy': 'total_energy',
                'forces': 'forces',
                'stress': 'virial',
                }
        super().__init__(context, config)

    def initialize(self, dataset: Dataset) -> None:
        assert self.config_future is None
        assert self.model_future is None
        self.deploy_future = {}
        logger.info('initializing {} using dataset of {} states'.format(
            self.__class__.__name__, dataset.length().result()))
        self.config_future = self.context.apps(NequIPModel, 'initialize')( # to initialized config
                self.config_raw,
                inputs=[dataset.data_future],
                outputs=[self.context.new_file('model_', '.pth')],
                )
        self.model_future = self.config_future.outputs[0] # to undeployed model

    def deploy(self) -> None:
        assert self.config_future is not None
        assert self.model_future is not None
        self.deploy_future['float32'] = self.context.apps(NequIPModel, 'deploy_float32')(
                self.config_future,
                inputs=[self.model_future],
                outputs=[self.context.new_file('deployed_', '.pth')],
                ).outputs[0]
        self.deploy_future['float64'] = self.context.apps(NequIPModel, 'deploy_float64')(
                self.config_future,
                inputs=[self.model_future],
                outputs=[self.context.new_file('deployed_', '.pth')],
                ).outputs[0]

    def save_deployed(
            self,
            path_deployed: Union[Path, str],
            dtype: str = 'float32',
            ) -> DataFuture:
        return copy_data_future(
                inputs=[self.deploy_future[dtype]],
                outputs=[File(str(path_deployed))],
                ).outputs[0] # return data future

    def set_seed(self, seed: int) -> None:
        self.config_raw['seed'] = seed
        self.config_raw['dataset_seed'] = seed

    @classmethod
    def create_apps(cls, context: ExecutionContext) -> None:
        training_label    = context[cls]['training_executor']
        training_device   = context[cls]['training_device']
        training_dtype    = context[cls]['training_dtype']
        training_walltime = context[cls]['training_walltime']

        model_label  = context[cls]['evaluate_executor']
        model_device = context[cls]['evaluate_device']
        model_ncores = context[cls]['evaluate_ncores']
        model_dtype  = context[cls]['evaluate_dtype']

        app_initialize = python_app(initialize, executors=[model_label], cache=False)
        context.register_app(cls, 'initialize', app_initialize)
        deploy_unwrapped = python_app(deploy, executors=[model_label], cache=False)
        def deploy_float32(config, inputs=[], outputs=[]):
            return deploy_unwrapped(
                    model_device,
                    'float32',
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'deploy_float32', deploy_float32)
        def deploy_float64(config, inputs=[], outputs=[]):
            return deploy_unwrapped(
                    model_device,
                    'float64',
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'deploy_float64', deploy_float64)
        train_unwrapped = python_app(train, executors=[training_label], cache=False)
        def train_wrapped(config, inputs=[], outputs=[]):
            return train_unwrapped(
                    training_device,
                    training_dtype,
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    walltime=training_walltime,
                    )
        context.register_app(cls, 'train', train_wrapped)
        evaluate_unwrapped = python_app(
                evaluate_dataset,
                executors=[model_label],
                cache=False,
                )
        def evaluate_wrapped(inputs=[], outputs=[]):
            return evaluate_unwrapped(
                    model_device,
                    model_dtype,
                    model_ncores,
                    cls.load_calculator,
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'evaluate', evaluate_wrapped)

    @classmethod
    def load_calculator(
            cls,
            path_model: Union[Path, str],
            device: str,
            dtype: str,
            set_global_options: str = 'warn',
            ) -> BaseCalculator:
        from nequip.ase import NequIPCalculator
        return NequIPCalculator.from_deployed_model(
                model_path=path_model,
                device=device,
                set_global_options=set_global_options,
                )


@id_for_memo.register(type(NequIPModel.load_calculator))
def id_for_memo_method(method, output_ref=False):
    string = inspect.getsource(method)
    return bytes(string, 'utf-8')
