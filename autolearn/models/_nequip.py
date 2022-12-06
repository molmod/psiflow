from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.models import BaseModel
from autolearn.execution import ModelExecutionDefinition, \
        TrainingExecutionDefinition


def get_elements(data):
    from ase.data import chemical_symbols
    _all = [set(a.numbers) for a in data]
    numbers = sorted(list(set(b for a in _all for b in a)))
    return [chemical_symbols[n] for n in numbers]


def to_nequip_dataset(data, nequip_config):
    import tempfile
    from nequip.utils import Config, instantiate
    from nequip.data.transforms import TypeMapper
    from nequip.data import ASEDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        nequip_config_dict = dict(nequip_config)
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
    return ase_dataset


def initialize(config, inputs=[], outputs=[]):
    import torch
    from nequip.utils import Config
    from nequip.scripts.train import default_config
    from nequip.model import model_from_config

    from autolearn.dataset import read_dataset
    from autolearn.models._nequip import to_nequip_dataset

    nequip_config = Config.from_dict(
            config,
            defaults=default_config,
            )
    ase_dataset = to_nequip_dataset(
            read_dataset(inputs=[inputs[0]]),
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


def deploy(device, dtype, nequip_config, inputs=[], outputs=[]):
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


def train(device, dtype, nequip_config, inputs=[], outputs=[]):
    import torch
    import tempfile
    from nequip.utils import Config
    from nequip.model import model_from_config
    from nequip.utils.versions import check_code_version
    from nequip.utils._global_options import _set_global_options
    from nequip.train.trainer import Trainer
    from autolearn.dataset import read_dataset
    from autolearn.models._nequip import to_nequip_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        nequip_config['device'] = device
        nequip_config['root'] = tmpdir
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

        check_code_version(nequip_config, add_to_config=True)
        _set_global_options(nequip_config)
        trainer = Trainer(model=None, **dict(nequip_config))
        training   = read_dataset(inputs=[inputs[1]])
        validation = read_dataset(inputs=[inputs[2]])
        trainer.n_train = len(training)
        trainer.n_val   = len(validation)

        data_train    = to_nequip_dataset(training, nequip_config)
        data_validate = to_nequip_dataset(validation, nequip_config)
        trainer.set_dataset(data_train, data_validate)
        trainer.model = model

        # Store any updated config information in the trainer
        trainer.update_kwargs(nequip_config)
        trainer.train()
    torch.save(trainer.model.to('cpu').state_dict(), outputs[0].filepath)


class NequIPModel(BaseModel):

    def __init__(self, context, config):
        self.config  = dict(config)
        self.config['device'] = 'cpu' # only use cuda if requested
        self.config.pop('include_keys', None)
        self.config.pop('key_mapping', None)
        self.config['dataset_include_keys'] = ['total_energy', 'forces', 'virial']
        self.config['dataset_key_mapping'] = {
                'energy_qm': 'total_energy', # custom key to ASE default key
                'forces_qm': 'forces',
                'stress_qm': 'virial',
                }
        self.context = context

        self.future        = None
        self.future_config = None
        self.future_deploy = None

    def initialize(self, dataset):
        assert self.future_deploy is None
        assert self.future_config is None
        assert self.future is None
        p_initialize = python_app(
                initialize,
                executors=[self.executor_label],
                )
        self.future_config = p_initialize(
                self.config,
                inputs=[dataset.future],
                outputs=[File(self.new_model())],
                )
        self.future = self.future_config.outputs[0]

    def deploy(self):
        device = self.context[ModelExecutionDefinition].device
        dtype  = self.context[ModelExecutionDefinition].dtype
        p_deploy = python_app(
                deploy,
                executors=[self.executor_label],
                )
        self.future_deploy = p_deploy(
                device,
                dtype,
                self.future_config,
                inputs=[self.future],
                outputs=[File(self.new_deploy())],
                ).outputs[0]

    def train(self, training, validation):
        self.future_deploy = None
        device = self.context[TrainingExecutionDefinition].device
        dtype  = self.context[TrainingExecutionDefinition].dtype
        executor_label = self.context[TrainingExecutionDefinition].executor_label
        p_train = python_app(
                train,
                executors=[executor_label],
                )
        self.future = p_train(
                device,
                dtype,
                self.future_config,
                inputs=[self.future, training.future, validation.future],
                outputs=[File(self.new_model())],
                ).outputs[0]

    @staticmethod
    def load_calculator(path_model, device, dtype):
        from nequip.ase import NequIPCalculator
        return NequIPCalculator.from_deployed_model(
                model_path=path_model,
                device=device,
                )
