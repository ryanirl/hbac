################################################################################
# A bunch of helper functions to enable more advanced argparse usage. 
#
# The idea behind the functionallity of these functions are easy: For any class,
# function, dataclass, etc we seperate out their parameters and break the parameters 
# themselved into a (possibly nested) dictionary containing four main parts: 
#
#  - Key: Parameter name
#  - Value: Type hint (if any provided)
#  - Value: Default value (if any provided)
#  - Value: Description (if any provided)
#
# Then given this standardization, we can easily add them to argparse in a
# recusive manner. Pydantic and yaml also makes advanced CLI argument parsing
# pretty easy.
################################################################################

import dataclasses
import warnings
import argparse
import yaml
import sys
import os

import inspect
from inspect import Parameter

from pydantic import ConfigDict
from pydantic.type_adapter import TypeAdapter

from omegaconf import OmegaConf
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf._utils import get_yaml_loader

from hbac.utils.config import parse_config
import hbac.utils.registry as registry

from typing import Sequence
from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple
from typing import Type
from typing import Dict
from typing import List
from typing import Any

_PYDANTIC_CONFIG_DICT = ConfigDict(arbitrary_types_allowed = True)
MISSING = "???"


def pre_parse_user_config(
    default_config: Optional[str] = None,
    *,
    aliases: Sequence[str] = ["-c", "--config"],
    argv: Optional[List[str]] = None,
    dest: str = "config"
) -> DictConfig:
    """In many of the config-based scripts it's advantageous to pre-parse the
    user input config before the main CLI setup. This is to allow functionality
    like the following:

        ``` $ python run.py -c config.yaml --print-config ```

    Or to dynamically modify custom configs that are not default:

        ``` $ python run.py -c config.yaml --arg value ```

    """
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument(
        *aliases, 
        default = default_config, 
        type = str,
        dest = dest
    )

    args, _ = parser.parse_known_args(argv)
    if (args.config is not None) and (not os.path.exists(args.config)):
        raise ValueError(f"The config path '{args.config}' does not exist!")

    config = parse_config(getattr(args, dest) or {})
    if isinstance(config, ListConfig):
        raise RuntimeError("The config must be a dictionary, not a list.")

    return config


def yaml_type_handler(f_type: object) -> Callable:
    """A cheeky way to do handle more advanced types in Argparse."""
    validator = TypeAdapter(f_type, config = _PYDANTIC_CONFIG_DICT)
    def caster(f_inp: str) -> Any:
        f_inp = _static_handler(f_inp)
        value = yaml.load(f_inp, Loader = get_yaml_loader())
        value = validator.validate_python(value)
        return value
    return caster


def pre_parse_field(name: str, default: Any = None, type: object = Any, **kwargs) -> Any:
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument(
        "--" + name, 
        default = default, 
        type = yaml_type_handler(type),
        **kwargs
    )
    return getattr(parser.parse_known_args()[0], name)


def add_register_group_to_parser(
    parser: argparse._ActionsContainer, 
    name: str,
    *, 
    default: Optional[Any] = None,
    exclude: List[str] = ["self"],
    **kwargs
) -> None:
    """Registers are added in as their own groups. Here, we define the functionality
    to be the following:
    
     $ python run.py --name key 
    
    Here, `name` is the name of the register being added (ex: 'optimizer') and key
    is the name (arbitrary of upper-lower case) of the item within the register to 
    use (ex: 'Adam'). When doing this, the parameters are Adam (for example) would
    be autoamtically added to the CLI as we pre-parse the user input to build up 
    the CLI (using `parser.parse_known_args()`).
    """
    register = registry.get_registry(name)

    # Check if the user provided config contains information to override the default. 
    config = pre_parse_user_config()
    if name in config:
        default = config[name].get("_target_", default)

    # Avoid conflicts when converting to a nested config.
    dest = f"{name}._target_"

    # Add the register as an argparse group.
    group = parser.add_argument_group(name)
    group.add_argument("--" + name, default = default, type = str, dest = dest, **kwargs) 

    # Here we dynamically pre-parse the users option to build up the config.
    registry_key = pre_parse_field(dest, default = default) 
    if registry_key is not None:
        add_class_to_parser(
            parser = group, 
            cls = register.get(registry_key), 
            prefix = name, 
            exclude = exclude
        )


def add_dataclass_to_parser(
    parser: argparse._ActionsContainer, 
    dataclass, 
    *, 
    prefix: Optional[str] = None,
    **kwargs
) -> None:
    """ """
    params = _unfold_dict(_get_dataclass_parameters(dataclass))
    _add_dot_to_parser(parser, params, prefix = prefix, **kwargs)


def add_class_to_parser(
    parser: argparse._ActionsContainer, 
    cls: object, 
    *, 
    prefix: Optional[str] = None, 
    exclude: List[str] = ["self"],
    **kwargs
) -> None:
    """ """
    params = _unfold_dict(_get_class_parameters(cls, exclude = exclude))
    _add_dot_to_parser(parser, params, prefix = prefix, **kwargs)


def add_function_to_parser(
    parser: argparse._ActionsContainer, 
    fn: Callable, 
    *, 
    prefix: Optional[str] = None, 
    exclude: List[str] = ["self"],
    **kwargs
) -> None:
    """ """
    params = _unfold_dict(_get_function_parameters(fn, exclude = exclude))
    _add_dot_to_parser(parser, params, prefix = prefix, **kwargs)


def parse_args(args: argparse.Namespace) -> DictConfig:
    """Simultaneously fold the dict and converting to OmegaConf."""
    config = OmegaConf.create({})
    for k, v in vars(args).items():
        if isinstance(v, tuple):
            v = list(v) # OmegaConf does NOT support tuples.
        OmegaConf.update(config, k, v)
    return config


################################################################################
# Utils
################################################################################


def _get_function_parameters(fn: Callable, *, exclude: List[str] = ["self"]) -> Dict[str, Tuple[Type, Any]]:
    """ """
    fields = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in exclude:
            continue

        f_type = param.annotation if param.annotation != Parameter.empty else Any
        f_default = param.default if param.default != Parameter.empty else MISSING

        fields[name] = (f_type, f_default)

    return fields


def _get_class_parameters(cls: object, *, exclude: List[str] = ["self"]) -> Dict[str, Tuple[Type, Any]]:
    return _get_function_parameters(cls.__init__, exclude = exclude) # type: ignore


def _get_dataclass_parameters(cls) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        default = f.default if f.default != dataclasses.MISSING else None

        if dataclasses.is_dataclass(f.type):
            fields[f.name] = _get_dataclass_parameters(f.type)
        else:
            fields[f.name] = (f.type, default)

    return fields


def _static_handler(f_inp: str) -> str:
    """Allows custom handling of a few edge cases."""
    if f_inp.lower() in ["none"]:
        f_inp = "null"

    return f_inp


def _unfold_dict(dictionary: Union[Dict, DictConfig], prefix: str = "", delimiter: str = ".") -> List[Tuple[Any, Any]]:
    """ """
    if isinstance(dictionary, DictConfig):
        container = OmegaConf.to_container(dictionary)
        if not isinstance(container, Dict):
            raise RuntimeError("`dictionary` must either be a Dict or an OmegaConf DictConfig.")
        dictionary = container

    items = []
    for key, value in dictionary.items():
        new_key = prefix + delimiter + str(key) if prefix else str(key)
 
        if isinstance(value, Dict):
            items.extend(_unfold_dict(value, new_key, delimiter))
        else:
            items.append((new_key, value))

    return list(items)


def _add_dot_to_parser(
    parser: argparse._ActionsContainer, 
    parameters: List[Tuple[str, Tuple[Type, Any]]], 
    *,
    prefix: Optional[str] = None,
    **kwargs
) -> None:
    """ """
    if (prefix is not None) and (prefix != ""):
        prefix = prefix + "."
    else:
        prefix = ""

    for f_name, (f_type, f_default) in parameters:
        parser.add_argument(
            f"--{prefix}{f_name}",
            default = f_default,
            required = True if f_default == MISSING else False,
            type = yaml_type_handler(f_type),
            **kwargs
        )


class PrintConfigAction(argparse.Action):
    """Action to print the final config. The final config is a function of all
    of the input parameters. That is, the user-defined conig, the defalt values,
    and the CLI arguments passed in. 
    """
    def __init__(self, *args, **kwargs) -> None:
        kwargs["dest"] = argparse.SUPPRESS
        kwargs["nargs"] = 0

        super(PrintConfigAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string = None) -> None:
        user_config = parse_config(getattr(namespace, "config") or {})
        dot_config = vars(namespace)
        for k, v in dot_config.items():
            if isinstance(v, tuple): 
                v = list(v) # OmegaConf does NOT support tuples.
            OmegaConf.update(user_config, k, v)

        sys.stdout.write(OmegaConf.to_yaml(user_config))
        sys.exit(0)


class ConfigAction(argparse.Action):
    """The config action is a very important action that allows the user to both
    load in a config, but it also gives the config prescidence over default
    values passed. 
    """
    def __init__(self, option_strings, dest, *args, **kwargs) -> None:
        if (option_strings != ["-c", "--config"]) or (dest != "config"):
            warnings.warn(
                f"To enable certain functionality, the option strings and dest in "
                f"`ConfigAction` are forced to be ['-c', '--config'] and 'config', "
                f"but '{option_strings}' and '{dest}' were found, and will be "
                f"overwritten. To remove this warning, please modify the add "
                f"arugment call to be `parser.add_argument('-c', '--config', ...)`"
            )

        # Force the option_strings and dest to be the following. This is
        # required to enable some of the functionality above. 
        super().__init__(["-c", "--config"], "config", *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string = None) -> None:
        config = pre_parse_user_config(aliases = self.option_strings)
            
        # Very important step to allow the config to hold prescedence over the
        # default parameters.
        for k, v in _unfold_dict(config):
            setattr(namespace, k, v)

        # Set the actual config arg with the path of the config. 
        setattr(namespace, self.dest, values)


