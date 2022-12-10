from parsl.app.app import python_app
from parsl.data_provider.files import File

from flower.models.base import evaluate_dataset, _new_deploy, _new_model
from flower.models import BaseModel
from flower.execution import ModelExecutionDefinition, \
        TrainingExecutionDefinition
from flower.utils import copy_data_future


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

    from flower.dataset import read_dataset
    from flower.models._nequip import to_nequip_dataset

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
    from flower.dataset import read_dataset
    from flower.models._nequip import to_nequip_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
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
        trainer.train()
    torch.save(trainer.model.to('cpu').state_dict(), outputs[0].filepath)


class NequIPModel(BaseModel):
    """Container class for NequIP models"""

    def __init__(self, context, config, dataset):
        super().__init__(context)

        self.config_future = self.context.apps(NequIPModel, 'initialize')( # to initialized config
                dict(config),
                inputs=[dataset.data_future],
                outputs=[File(_new_model(context))],
                )
        self.model_future  = self.config_future.outputs[0] # to undeployed model
        self.deploy_future = None # to deployed model

    def deploy(self):
        self.deploy_future = self.context.apps(NequIPModel, 'deploy')(
                self.config_future,
                inputs=[self.model_future],
                outputs=[File(_new_deploy(self.context))],
                ).outputs[0]

    def train(self, training, validation):
        self.deploy_future = None # no longer valid
        self.model_future  = self.context.apps(NequIPModel, 'train')( # new DataFuture instance
                self.config_future,
                inputs=[self.model_future, training.data_future, validation.data_future],
                outputs=[File(_new_model(self.context))]
                ).outputs[0]

    def save_deployed(self, path_deployed):
        return copy_data_future(
                inputs=[self.deploy_future],
                outputs=[File(str(path_deployed))],
                )

    @classmethod
    def create_apps(cls, context):
        training_label  = context[TrainingExecutionDefinition].executor_label
        training_device = context[TrainingExecutionDefinition].device
        training_dtype  = context[TrainingExecutionDefinition].dtype

        model_label  = context[ModelExecutionDefinition].executor_label
        model_device = context[ModelExecutionDefinition].device
        model_dtype  = context[ModelExecutionDefinition].dtype
        model_ncores = context[ModelExecutionDefinition].ncores

        app_initialize = python_app(initialize, executors=[model_label])
        context.register_app(cls, 'initialize', app_initialize)
        deploy_unwrapped = python_app(deploy, executors=[model_label])
        def deploy_wrapped(config, inputs=[], outputs=[]):
            return deploy_unwrapped(
                    model_device,
                    model_dtype,
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'deploy', deploy_wrapped)
        train_unwrapped = python_app(train, executors=[training_label])
        def train_wrapped(config, inputs=[], outputs=[]):
            return train_unwrapped(
                    training_device,
                    training_dtype,
                    config,
                    inputs=inputs,
                    outputs=outputs,
                    )
        context.register_app(cls, 'train', train_wrapped)
        #app_copy_model = python_app(copy_file, executors=[model_label])
        #context.register_app(cls, 'copy_model', app_copy_model)
        evaluate_unwrapped = python_app(evaluate_dataset, executors=[model_label])
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
    def load_calculator(cls, path_model, device, dtype):
        from nequip.ase import NequIPCalculator
        return NequIPCalculator.from_deployed_model(
                model_path=path_model,
                device=device,
                )
