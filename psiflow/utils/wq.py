# TODO: this probably does not work in a nested way
#  or when join apps submit tasks at some point in the future (after exiting the context)

import logging


logger = logging.getLogger(__name__)  # logging per module


WQ_RESOURCES_REGISTRY = []


def register_definition(definition: 'ExecutionDefinition') -> None:
    """"""
    if (spec := definition.spec) is None:
        return  # threadpool does not have priority

    WQ_RESOURCES_REGISTRY.append((definition.name, spec))
    spec["priority"] = SetWQPriority.default


class SetWQPriority:
    """Manage the WQ priority tag as context manager"""
    default = 0

    def __init__(self, value: int, verbose: bool = False) -> None:
        self.value = value
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            logger.info(f"SetWQPriority setting priority:\t{self.value}")
        for n, spec in WQ_RESOURCES_REGISTRY:
            spec["priority"] = self.value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            logger.info(f"SetWQPriority unsetting {self.value}")
        for n, spec in WQ_RESOURCES_REGISTRY:
            spec["priority"] = SetWQPriority.default
