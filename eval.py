from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import argparse
import logging

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hbac.utils.trainer as T
import hbac.utils.cli as cli
from hbac.datasets.hms import HmsDataModule
from hbac.utils.logger import setup_logger
from hbac.utils.utils import get_default_device
from hbac.registry import instantiate

from train import HbacMultimodalTask
from train import HbacUnimodalTask

from typing import Optional
from typing import Tuple
from typing import Dict
from typing import List
from typing import Any

logger = logging.getLogger("eval")


class HbacEvaluator:
    def __init__(self, task: T.Task, device: Optional[str] = None) -> None:
        self.task = task
        self.device = device if device else get_default_device()

        # Register the evaluator with the task.
        self.task.callbacks.append(self)

        # Storage of the metrics througout the run of the evaluation.
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    def log(self, k: str, v: float, to_pbar: bool = True) -> None:
        self.metrics[k].append(v)

    def _log_dict(self, msg: Dict, prefix: str = "") -> None:
        logger.info(prefix + " | ".join([f"{k}: {v}" for k, v in msg.items()]))

    def to_device(self, batch: Tuple[Any]) -> Tuple[Any]:
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(self.device)

        return batch

    def run(self, model: nn.Module, val_dataloader: DataLoader) -> None:
        model = model.to(self.device)
        max_steps = len(val_dataloader)

        # Flush the metrics upon the start of eval.
        self.metrics.clear()

        model.eval()
        with torch.no_grad():
            logger.info("Starting validation")
            for step, batch in tqdm(enumerate(val_dataloader, 1), total = max_steps):
                batch = self.to_device(batch)
                self.task.validation_step(model, batch)

        for k, v in self.metrics.items():
            values = np.array(v).reshape(-1)
            self._log_dict({
                "mean": f"{np.mean(values):.6f}",
                "median": f"{np.median(values):.6f}",
                "min": f"{np.min(values):.6f}",
                "max": f"{np.max(values):.6f}",
                "std": f"{np.std(values):.6f}",
            }, prefix = f"Summary [{k}] - ")


def validate_config_field(args: argparse.Namespace, field: str) -> Any:
    if field not in args:
        raise ValueError(f"Expected the config to have field '{field}'.")

    return getattr(args, field)


def parse_args(argv: Optional[List[str]] = None) -> DictConfig:
    name = "evaluation"
    desc = "This script enables a more comprehensive evaluation" 

    parser = argparse.ArgumentParser(prog = name, description = desc)
    parser.add_argument(
        "-m", "--mode", type = str, default = "unimodal", 
        choices = ["unimodal", "multimodal"], metavar = ""
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--model-path", type = str, required = True, metavar = "", 
        help = "The path to the '.pt' model file."
    )
    required.add_argument(
        "-c", "--config", type = str, action = cli.ConfigAction, metavar = ""
    )

    args = parser.parse_args(argv)
    cfg = cli.parse_args(args)

    # Parse and validate the config input. 
    validate_config_field(args, "data.data_dir")
    validate_config_field(args, "data.data_type")
    validate_config_field(args, "data.fold")
    validate_config_field(args, "data.n_folds")
    validate_config_field(args, "model._target_")

    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Setup the logger, no file handler here for now. 
    setup_logger()

    logger.info(f"Instantiating validation dataloader")
    val_dataloader = HmsDataModule(**args.data).get_validation_loader()

    logger.info("Instantiating Model")
    model = instantiate("model", args.model)
    logger.info(model)
    logger.info(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    logger.info(f"Loading model checkpoint '{args.model_path}'.")
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])

    logger.info(f"Evuating with the {args.mode} task")
    if args.mode == "unimodal": task: T.Task = HbacUnimodalTask()
    elif args.mode == "multimodal": task = HbacMultimodalTask()
    else: raise ValueError("mode must be one of (unimodal, multimodal).")

    logger.info("Starting evaluation!")
    evaluator = HbacEvaluator(task)
    evaluator.run(model, val_dataloader)


if __name__ == "__main__":
    main()


