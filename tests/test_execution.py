import yaml

from psiflow.execution import ExecutionDefinition


def test_execution():
    """Check a few execution parameters. This could definitely be expanded upon."""
    data_yaml = """
    executor: threadpool
    max_threads: 42
    use_gpu: false
    max_runtime: 00:59:00
    local:
      cores: 4
      memory: 8
    """
    definition = ExecutionDefinition.from_config(**yaml.safe_load(data_yaml))
    assert definition.executor_type == "threadpool"
    assert definition.provider is None
    assert definition.container is None
    assert definition.lifetime == float("inf")
    assert definition.max_runtime == 3540
    assert definition.task_slots == 42
    assert definition.cores_per_task == 1
    assert definition.use_gpu == False
    assert definition.spec is None

    data_yaml = """
    cores_per_task: 3
    gpus_per_task: 0
    mem_per_task: 6
    min_runtime: 00:15:00
    env_vars:
      CUSTOM_KEY: custom_var
    slurm:
      cores_per_node: 8
      mem_per_node: 16
      walltime: "02:00:00"
    """
    spec = {
        "cores": 3,
        "disk": 0,
        "gpus": 0,
        "memory": 6000,
        "priority": 0,
        "running_time_min": 900,
    }
    data = yaml.safe_load(data_yaml)
    definition = ExecutionDefinition.from_config(**data)
    assert definition.executor_type == "workqueue"
    assert definition.provider is not None
    assert definition.lifetime == 7200
    assert definition.max_runtime == 7140
    assert definition.task_slots == 2
    assert definition.cores_per_task == 3
    assert definition.use_gpu == False
    assert definition.spec == spec
    assert definition.env_vars["CUSTOM_KEY"] == "CUSTOM_VAR"

