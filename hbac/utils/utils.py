import torch
import glob
import os

from omegaconf import OmegaConf
from omegaconf import DictConfig


class OutputVersioning:
    def __init__(self, base: str, prefix: str = "version") -> None:
        self.base = base
        self.prefix = prefix

        if not os.path.exists(self.base):
            raise ValueError(f"The input base directory '{base}' must already exist.")

    def create_new_output(self) -> str:
        output_dir = os.path.join(self.base, self._get_next_dir())
        assert not os.path.exists(output_dir)
        assert os.path.exists(self.base)
        os.makedirs(output_dir)
        return output_dir

    def _get_next_dir(self) -> str:
        taken = []
        for file in glob.glob(os.path.join(self.base, f"{self.prefix}_*")):
            if not os.path.isdir(file):
                continue 

            words = file.split("_")
            if len(words) != 2:
                continue
            
            try:
                taken.append(int(words[1]))
            except ValueError:
                continue

        taken = sorted(taken)
        if len(taken) == 0:
            value = 0
        else:
            value = taken[-1] + 1
        
        return f"{self.prefix}_{value}"


def get_default_device() -> str:
    """
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pretrained(path: str, model: torch.nn.Module):
    """
    """
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model


def save_config(path: str, config: DictConfig) -> None:
    """
    """
    OmegaConf.save(config, path)


