import os

from omegaconf import OmegaConf
from omegaconf import DictConfig
from omegaconf import ListConfig

from typing import Union
from typing import Dict


def parse_config(cfg: Union[str, Dict, DictConfig]) -> Union[DictConfig, ListConfig]:
    """This function takes in any reasonable type and tries to convert it to an
    OmegaConf config. It will accept file paths, yaml strings, and dictionaries.

    Args:
        cfg (Union[str, Dict, DictConfig]): The input to convert to an OmegaConf
            config.

    Returns:
        DictConfig: The OmegaConf config.

    """
    if isinstance(cfg, DictConfig):
        return cfg

    elif isinstance(cfg, Dict):
        return OmegaConf.create(cfg)

    elif isinstance(cfg, str):
        # This handles the case in which 'cfg' is a string. In this case it can
        # either be a file type, or a YAML string.
        if os.path.isfile(cfg):
            return OmegaConf.load(cfg)
        else: 
            return OmegaConf.create(cfg)

    else: 
        raise ValueError(
            f"The config '{cfg}' could not be loaded into an OmegaConf config. "
            f"Make sure the input is either a config, dictionary, yaml string, "
            f"or a valid string path to a file."
        )


