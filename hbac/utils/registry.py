import warnings
import inspect
import torch

from typing import Dict
from typing import Type
from typing import Any

_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "optimizer": {},
    "scheduler": {},
    "criterion": {} 
}


def add_registry(name: str, registry: Dict[str, Any]) -> None:
    if name in _REGISTRIES:
        warnings.warn( 
            f"The register '{name}' already exists, and will be overwritten. "
            f"Ensure that this is the expected behavior."
        )

    _REGISTRIES[name] = registry


def register(registry: str, k: str, v: Any) -> None:
    _registry = _REGISTRIES[registry]
    if k in _registry:
        warnings.warn( 
            f"The register '{registry}' already contains the key '{k}', and it "
            f"will be overwritten. Ensure that this is the expected behavior."
        )

    _registry[k] = v


def instantiate(_registry: str, _params: Dict, **kwargs) -> Any:
    target = _params["_target_"]
    _params = {k: v for k, v in _params.items() if k != "_target_"}
    _params.update(kwargs)
    return _REGISTRIES[_registry][target](**_params)


def get_registry(name: str) -> Dict[str, Any]:
    return _REGISTRIES[name]


################################################################################
# Utils
################################################################################


def _get_default_subclasses(search: object, object_types: Any) -> Dict[str, Type]:
    if not isinstance(object_types, list):
        object_types = [object_types]

    subclasses = {}
    for name, obj in inspect.getmembers(search):
        if not inspect.isclass(obj):
            continue

        # It must subclass one of the object_types, but it cannot be any of the
        # object_types themselves.
        is_subclass = any([issubclass(obj, obj_type) for obj_type in object_types])
        is_one_of = any([obj == obj_type for obj_type in object_types])
        if is_subclass and not is_one_of:
            subclasses[name] = obj

    return subclasses


def _populate_torch_registries() -> None:
    # This method does not capture everything, and is far from perfect. But I
    # haven't needed to update it yet. 
    t_OPTIMIZER = torch.optim.Optimizer
    t_CRITERION = torch.nn.modules.loss._Loss
    t_SCHEDULER = [
        torch.optim.lr_scheduler.LRScheduler,  # > torch 2.0
        torch.optim.lr_scheduler._LRScheduler, # < torch 2.0
        torch.optim.lr_scheduler.ReduceLROnPlateau
    ]

    for name, cls in _get_default_subclasses(torch.optim, t_OPTIMIZER).items():
        register("optimizer", name, cls)

    for name, cls in _get_default_subclasses(torch.nn, t_CRITERION).items():
        register("criterion", name, cls)

    for name, cls in _get_default_subclasses(torch.optim.lr_scheduler, t_SCHEDULER).items():
        register("scheduler", name, cls)


_populate_torch_registries()


if __name__ == "__main__":
    # Print out the default registries. 
    for k, v in get_registry("optimizer").items(): print(k, v)
    print("\n" + ("-" * 25) + "\n")
    for k, v in get_registry("criterion").items(): print(k, v)
    print("\n" + ("-" * 25) + "\n")
    for k, v in get_registry("scheduler").items(): print(k, v)


