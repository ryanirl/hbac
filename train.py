import argparse
import logging
import math
import os

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hbac.utils.trainer as T
import hbac.utils.cli as cli
from hbac.datasets.hms import HmsDataModule
from hbac.utils.logger import setup_logger
from hbac.registry import instantiate

from typing import Optional
from typing import Tuple
from typing import List
from typing import Any

logger = logging.getLogger("train")


class HbacUnimodalTask(T.Task):
    """Defines the core logic for how to train a unimodal model for the HMS-HBAC 
    Kaggle competition. Because we are tasked with learning the distribution of 
    expert annotator votes the KL-Divergence loss function is used. 
    
    Also the distribution of the number of expert annotators is bimodal and can
    be classified into two groups. The first group (l10) is where there are
    stricly less than 10 expert annotators, these are less confident samples.
    The other group contains greater than or equal to 10 expert annotators,
    these are 'high quality' samples. That is why it's seperated out in the
    validation step.
    """
    def __init__(self) -> None:
        super().__init__()

        self.train_criterion = nn.KLDivLoss(reduction = "batchmean")
        self.valid_criterion = nn.KLDivLoss(reduction = "none")

    def training_step(self, model: nn.Module, batch: Tuple[Any]) -> torch.Tensor:
        """Code for training the model one step."""
        x = batch[0]
        y = batch[1]

        y_hat = model(x)
        loss = self.train_criterion(y_hat, y)

        self.log("train/loss", loss.item())

        return loss 

    def validation_step(self, model: nn.Module, batch: Tuple[Any]) -> Any:
        """Code for evaluating the model one step."""
        x = batch[0]
        y = batch[1]
        count = batch[2]

        y_hat = model(x)
        loss = self.valid_criterion(y_hat, y).sum(axis = 1)

        # Split based on based >9 or <9.
        loss_all = loss.mean().item()
        loss_g10 = loss[count >= 9].mean().item()
        loss_l10 = loss[count < 9].mean().item()

        # Possible that there were no count less than or greater than 9.
        loss_g10 = loss_g10 if not math.isnan(loss_g10) else 0.0
        loss_l10 = loss_l10 if not math.isnan(loss_l10) else 0.0

        # Update the metrics 
        self.log_dict({
            "val/loss_all": loss_all,
            "val/loss_g10": loss_g10,
            "val/loss_l10": loss_l10
        })


class HbacMultimodalTask(T.Task):
    """Defines the core logic for how to train the final mulitmodal model for
    the HMS-HBAC Kaggle competition. For more information on the training task, 
    see the unimodal training task documentation. 
    """
    def __init__(self) -> None:
        super().__init__()

        self.train_criterion = nn.KLDivLoss(reduction = "batchmean")
        self.valid_criterion = nn.KLDivLoss(reduction = "none")

    def training_step(self, model: nn.Module, batch: Tuple[Any]) -> torch.Tensor:
        """Code for training the model one step."""
        eeg = batch[0]
        eeg_spectrogram = batch[-2]
        spectrogram = batch[-1]
        y = batch[1]

        y_hat = model(eeg, eeg_spectrogram, spectrogram)
        loss = self.train_criterion(y_hat, y)

        self.log("train/loss", loss.item())

        return loss 

    def validation_step(self, model: nn.Module, batch: Tuple[Any]) -> Any:
        """Code for evaluating the model one step."""
        eeg = batch[0]
        eeg_spectrogram = batch[-2]
        spectrogram = batch[-1]
        y = batch[1]
        count = batch[2]

        y_hat = model(eeg, eeg_spectrogram, spectrogram)
        loss = self.valid_criterion(y_hat, y).sum(axis = 1)

        # Split based on based >10 or <10.
        loss_all = loss.mean().item()
        loss_g10 = loss[count >= 9].mean().item()
        loss_l10 = loss[count < 9].mean().item()

        # Possible that there were no count less than or greater than 9.
        loss_g10 = loss_g10 if not math.isnan(loss_g10) else 0.0
        loss_l10 = loss_l10 if not math.isnan(loss_l10) else 0.0

        # Update the metrics 
        self.log_dict({
            "val/loss_all": loss_all,
            "val/loss_g10": loss_g10,
            "val/loss_l10": loss_l10
        })


def train(
    task: T.Task,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    trainer_args: T.TrainerArgs,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
) -> None:
    """
    """
    trainer = T.Trainer(
        args = trainer_args,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        task = task,
        callbacks = [
            T.ModelCheckpoint(monitor = "val/loss_g10", mode = "min"),
            T.ModelCheckpoint(monitor = "val/loss_l10", mode = "min"),
            T.ModelCheckpoint(monitor = "val/loss_all", mode = "min")
        ]
    )
    trainer.fit(
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Validate that the output directory does not already exist.
    output_dir = args.trainer.output_dir
    if os.path.exists(output_dir):
        raise FileExistsError(f"The directory '{output_dir}' already exists!")
    else:
        os.makedirs(output_dir, exist_ok = False)

    # Setup the logger, add a file handler to save the output.
    setup_logger(filename = os.path.join(output_dir, "train.log"))

    # Save the config to the output_dir.
    logger.info(f"Config: \n{OmegaConf.to_yaml(args)}")
    logger.info(f"Saving config to '{output_dir}'")
    OmegaConf.save(args, os.path.join(output_dir, "config.yaml"))

    logger.info("Instantiating TrainerArgs")
    trainer_args = T.TrainerArgs(**args.trainer)

    logger.info("Instantiating Model")
    model = instantiate("model", args.model)
    logger.info(model)
    logger.info(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    logger.info("Instantiating Optimizer")
    optimizer = instantiate("optimizer", args.optimizer, params = model.parameters())

    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if args.scheduler._target_ is not None:
        logger.info("Instantiating Scheduler")
        scheduler = instantiate("scheduler", args.scheduler, optimizer = optimizer)

    logger.info("Instantiating Dataloaders")
    train_dataloader, val_dataloader = HmsDataModule(**args.data).get_loaders()

    # Handle the `from_pretrained` option. 
    if args.from_pretrained is not None:
        if args.trainer.resume_from_checkpoint is not None:
            logger.warning(
                "Both 'from_pretrained' and 'from_checkpoint' are set. Ignoring "
                "'from_pretrained' and resuming training from checkpoint."
            )
        else:
            logger.info(f"Resuming from checkpoint '{args.from_pretrained}'.")
            model.load_state_dict(torch.load(args.from_pretrained)["model_state_dict"])

    logger.info(f"Training with the {args.mode} task")
    if args.mode == "unimodal": task: T.Task = HbacUnimodalTask()
    elif args.mode == "multimodal": task = HbacMultimodalTask()
    else: raise ValueError("mode must be one of (unimodal, multimodal).")

    logger.info("Starting training!")
    train(
        task = task,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        trainer_args = trainer_args,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader
    )


def parse_args(argv: Optional[List[str]] = None) -> DictConfig:
    """ """
    parser = argparse.ArgumentParser(
        add_help = False, formatter_class = argparse.RawDescriptionHelpFormatter
    )

    # `add_register_group_to_parser` will automatically check the user provided 
    # config
    cli.add_register_group_to_parser(
        parser = parser, 
        name = "optimizer", 
        default = "Adam",
        exclude = ["self", "params"]
    )
    cli.add_register_group_to_parser(
        parser = parser, 
        name = "scheduler", 
        default = None,
        exclude = ["self", "optimizer"]
    )
    cli.add_register_group_to_parser(
        parser = parser, 
        name = "model", 
        default = "eeg_cnn_rnn_att_base",
        exclude = ["self"]
    )
    cli.add_dataclass_to_parser(
        parser = parser.add_argument_group("trainer"),
        dataclass = T.TrainerArgs,
        prefix = "trainer"
    )
    cli.add_dataclass_to_parser(
        parser = parser.add_argument_group("data"),
        dataclass = HmsDataModule,
        prefix = "data"
    )

    # Default arguments
    parser.add_argument("-h", "--help", action = argparse._HelpAction)
    parser.add_argument("-p", "--print_config", action = cli.PrintConfigAction)
    parser.add_argument("-c", "--config", type = str, action = cli.ConfigAction) 
    parser.add_argument("-m", "--mode", type = str, default = "unimodal", choices = ["unimodal", "multimodal"], metavar = "")
    parser.add_argument("--from_pretrained", type = str, default = None)

    args = parser.parse_args(argv)
    config = cli.parse_args(args) # Dot-to-dict + conversion to OmegaConf

    return config


if __name__ == "__main__":
    main()


