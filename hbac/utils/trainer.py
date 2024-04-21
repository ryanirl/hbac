################################################################################
# A sort of header file containing a lightweight PyTorch Trainer class, similar 
# to FairSeq, Lightning, and HuggingFace's Trainer class in it's functionality, 
# but with a few differences in its abstraction.
# 
# Originally designed for rapid experimentation and 100% extendability to speed
# up experimentation in Kaggle competitions.
#
# The goal of this file is to be a bug-free (one can only hope) and sub 1000 
# line file that can easily be copy and pasted into pre existing libraries to 
# reduce dependencies speed up experimentation. 
#
# Requirements:
#  - torch
#  - tqdm
#
# Possible plans for future updates:
# - The method for which data is moved to device needs work.
# - `testing_step` has limited support.
# - Enable distributed training. 
# - I don't like how default Callbacks are handled. It makes it very hard to 
#   change this behavior from the user standpoint, for example if they wanted to
#   remove TQDMLogger and implement a custom version of it.
# - I don't like how the scheduler step is forced to be epoch-level. Sometimes
#   step-level is required, for example when you have very large datasets and 
#   train with a small epoch number. 
# - The way `output_dir` is handling isn't strict and is subject to change in 
#   the future. `Trainer` does nothing to handle or verify that the `output_dir`
#   either exists or does not exist, but callbacks such as `MetricTracker` 
#   requires that on train start both the `output_dir` exists and that
#   `{output_dir}/metrics/` does not exist to avoid overwriting data.
# - Would it make more sense to have model passed to the trainer in .fit() and 
#   not hold a reference to it? This would complicate saving checkpoints, but
#   a trainer checkpoint != the model checkpoint. 
#
# Author: Ryan "RyanIRL" Peters
#
# License: MIT 
#
################################################################################

from __future__ import annotations

from collections import defaultdict
import dataclasses
import unicodedata
import logging
import time
import csv
import gc
import os
import re

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional
from typing import Union
from typing import Dict
from typing import List
from typing import Any

__version__ = "0.1.0"

logger = logging.getLogger("trainer")


################################################################################
# Trainer
################################################################################

class TrainerCallbacks:
    """ Training hooks in order of when they are run. """
    def __init__(self):
        self.callbacks = []

    def on_train_start(self) -> None:
        for callback in self.callbacks:
            callback.on_train_start(self)

    def on_train_epoch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_train_epoch_start(self)

    def on_train_step_start(self, batch: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_step_start(self, batch)

    def on_train_step_end(self, output: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_step_end(self, output)

    def on_train_epoch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_epoch_end(self)

    def on_val_epoch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_val_epoch_start(self)

    def on_val_step_start(self, batch: Any) -> None:
        for callback in self.callbacks:
            callback.on_val_step_start(self, batch)

    def on_val_step_end(self, output: Any) -> None:
        for callback in self.callbacks:
            callback.on_val_step_end(self, output)

    def on_val_epoch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_val_epoch_end(self)

    def on_epoch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(self)

    def on_epoch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end(self)

    def on_keyboard_interrupt(self) -> None:
        for callback in self.callbacks:
            callback.on_keyboard_interrupt(self)

    def on_exception(self, e: Exception) -> None:
        for callback in self.callbacks:
            callback.on_exception(self, e)

    def log(self, k: str, v: float, to_pbar: bool = True) -> None:
        for callback in self.callbacks:
            callback.log(self, k, v, to_pbar)


@dataclasses.dataclass
class TrainerArgs:
    device: Optional[str] = None
    min_epochs: int = 0
    max_epochs: int = 50
    grad_accum_steps: int = 1
    check_val_every_n_epochs: int = 1
    log_every_n_steps: int = 1
    resume_from_checkpoint: Optional[str] = None
    output_dir: str = "./output/"


@dataclasses.dataclass
class TrainerState:
    mode: Optional[str] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    curr_epoch: int = 0
    curr_step: int = 0
    is_training: bool = False
    should_stop: bool = False


class Trainer(TrainerCallbacks):
    def __init__(
        self,
        task: Task,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        callbacks: Union[Callback, List[Callback]] = [],
        args: Optional[TrainerArgs] = None
    ) -> None:
        """
        """
        super().__init__()

        self.task = task 
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args if args is not None else TrainerArgs()
        self.state = TrainerState() 

        # Handle and verify some of the input arguments.
        self.task.callbacks.append(self)
        self.args.device = self.args.device if self.args.device is not None else get_default_device()

        # Handle the callbacks.
        self.callbacks: List[Callback] = [
            ToDevice(self.args.device), 
            TQDMLogger(), 
            MetricTracker(), 
            SaveFinalModel(), 
            SummaryWriter()
        ]
        if not isinstance(callbacks, List):
            callbacks = [callbacks]
        self.callbacks.extend(callbacks)

        # Load the trainer from checkpoint if needed. 
        if self.args.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint at '{self.args.resume_from_checkpoint}'")
            self.load_state_dict(
                torch.load(self.args.resume_from_checkpoint, map_location = "cuda"), 
                resume_from_checkpoint = True
            )

    def load_state_dict(self, state_dict: Dict[str, Any], resume_from_checkpoint: bool = False) -> None:
        new_args = TrainerArgs(**state_dict["trainer_args"])
        new_args.output_dir = self.args.output_dir
        self.args = new_args

        self.state = TrainerState(**state_dict["trainer_state"])
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        if self.scheduler is not None: 
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        for state, callback in zip(state_dict["callbacks_state_dict"], self.callbacks):
            callback.load_state_dict(state)

        if resume_from_checkpoint: # This logic is a work in progress.
            self.args.min_epochs = self.state.curr_epoch

    def state_dict(self) -> Dict[str, Any]:
        # Trainer specific state. Notice that we do NOT hold a reference to the model.
        trainer_state = {
            "trainer_args": dataclasses.asdict(self.args),
            "trainer_state": dataclasses.asdict(self.state),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "callbacks_state_dict": [callback.state_dict() for callback in self.callbacks]
        }

        return trainer_state

    def save_checkpoint(self, filename: str) -> None:
        torch.save(self.state_dict(), os.path.join(self.args.output_dir, filename))

    def close(self):
        """Clears references to optimizers, schedulers, and model."""
        self.model = None # type: ignore
        self.optimizer = None # type: ignore
        self.scheduler = None 

        # Clear any other references to tensors.
        torch.cuda.empty_cache() # If using CUDA, release GPU memory.
        gc.collect()

    def _training_step(self, batch: Any) -> torch.Tensor:
        self.on_train_step_start(batch)

        # What the user implements is a closure for the optimizer. This allows certain 
        # algorithms like LBFGS to run the loop multiple times. It also gives the user
        # a bit more flexibility in how they define this loop. Read more about closures
        # here: https://pytorch.org/docs/stable/optim.html#optimizer-step-closure. Note 
        # that the trainer will handle the optimizer steps. We modify this training step
        # with the `on_train_step_start` and `on_train_step_end` hooks. 
        output = self.task.training_step(self.model, batch)
        self.on_train_step_end(output)

        # The user can return either a dict or a torch.Tensor loss.
        loss = output if isinstance(output, torch.Tensor) else output["loss"]
        loss.backward()

        return loss

    def _train_loop(self, train_dataloader: DataLoader) -> None:
        """ """
        torch.set_grad_enabled(True)
        self.model.train()
        self.state.mode = "train"

        self.on_train_epoch_start()
        for self.state.curr_step, batch in enumerate(train_dataloader, 1):
            self.optimizer.step(lambda: self._training_step(batch)) # type: ignore

            # Whether or not to accumulate the gradients. 
            if self.state.curr_step % self.args.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none = True)

        if self.scheduler is not None: 
            self.scheduler.step()

        self.on_train_epoch_end()
        self.state.mode = None

    def _validation_loop(self, val_dataloader: DataLoader) -> None:
        torch.set_grad_enabled(False)
        self.model.eval()
        self.state.mode = "validation"

        self.on_val_epoch_start()
        for self.state.curr_step, batch in enumerate(val_dataloader, 1):
            self.on_val_step_start(batch)
            output = self.task.validation_step(self.model, batch)
            self.on_val_step_end(output)

        self.on_val_epoch_end()
        self.state.mode = None

        # Enable gradients + batchnorm + dropout.
        torch.set_grad_enabled(True)

        self.model.train()

    def _fit(self,train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:
        """Main training loop."""
        self.max_train_steps = len(train_dataloader)
        self.max_val_steps = -1 if val_dataloader is None else len(val_dataloader)

        # Only move the model to device during training. 
        self.model = self.model.to(self.args.device)

        self.on_train_start()
        for self.state.curr_epoch in range(self.args.min_epochs, self.args.max_epochs + 1):
            self.on_epoch_start()

            # Always skip training on first epoch and run evaluation. 
            if self.state.curr_epoch != 0:
                self._train_loop(train_dataloader)

            # Whether or not to every the validation loop. Always do validation first (see above).
            if (self.state.curr_epoch % self.args.check_val_every_n_epochs == 0) and (val_dataloader is not None):
                self._validation_loop(val_dataloader)

            # Allows callbacks to exit the training loop (ex: EarlyStopping).
            if self.state.should_stop:
                break

            self.on_epoch_end()
        self.on_train_end()

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Trainer:
        """ """
        self.state.is_training = True

        # Wrap the main training loop in a try-except to catch and gracefuly exit keyboard interupts.
        try:
            self._fit(train_dataloader, val_dataloader)

        except KeyboardInterrupt:
            logging.warning(f"KeyboardInterupt raised! Attempting to exit gracefuly...")
            self.on_keyboard_interrupt()

        except Exception as e:
            logging.error(f"Expetion raised: {e}")
            logging.error(e, exc_info = True)
            self.on_exception(e)

        self.state.is_training = False
        return self


################################################################################
# Base Task
################################################################################

class Task:
    """A `Task` is a class that defines the logic for training a specific model
    in PyTorch. It has similar features to Lightning's LightningModule, but it
    doesn't hold a reference to the model itself, and instead defines how to use
    this model to perform a training, validation, and testing step given a batch
    from a dataloader.

    Under this abstraction it defines a 'training task' rather than another 
    nn.Module abstraction like Lightning does. I personally believe this makes
    more sense. 

    The only required method to overload is `training_step()`, but if you want to
    perform validation or testing then `validation_step()` and `testing_step()`
    must be overwritten.
    """
    def __init__(self):
        # The callbacks for a `Task` are things that directly exploit the task itelf
        # such as `Trainer` or an evaluator for example. For example, any class that 
        # needs access to `training_step()` or the logging information. 
        self.callbacks: List[Any] = []

    def training_step(self, model: nn.Module, batch: Any) -> Union[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError()

    def validation_step(self, model: nn.Module, batch: Any) -> Any:
        raise NotImplementedError()

    def testing_step(self, model: nn.Module, batch: Any) -> Any:
        raise NotImplementedError()

    def log(self, k: str, v: Union[float, torch.Tensor], to_pbar: bool = True) -> None:
        if isinstance(v, torch.Tensor): 
            v = v.item()

        for callback in self.callbacks:
            callback.log(k, v, to_pbar)

    def log_dict(self, kv: Dict, to_pbar: bool = True) -> None:
        for k, v in kv.items():
            self.log(k, v, to_pbar)


################################################################################
# Callbacks
################################################################################

class Callback:
    """ Training hooks in order of when they are run. """
    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_epoch_start(self, trainer: Trainer) -> None:
        pass

    def on_train_epoch_start(self, trainer: Trainer) -> None:
        pass

    def on_train_step_start(self, trainer: Trainer, batch: Any) -> None:
        pass

    def on_train_step_end(self, trainer: Trainer, output: Any) -> None:
        pass

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        pass

    def on_val_epoch_start(self, trainer: Trainer) -> None:
        pass

    def on_val_step_start(self, trainer: Trainer, batch: Any) -> None:
        pass

    def on_val_step_end(self, trainer: Trainer, output: Any) -> None:
        pass

    def on_val_epoch_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_keyboard_interrupt(self, trainer: Trainer) -> None:
        pass

    def on_exception(self, trainer: Trainer, e: Exception) -> None:
        pass

    def log(self, trainer: Trainer, k: str, v: float, to_pbar: bool = True) -> None:
        pass

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}


class ToDevice(Callback):
    """
    """
    def __init__(self, device: Optional[str] = None) -> None:
        # Track this here for state_dict(), will be updated when `on_train_start` is run.
        self.device = device

    def on_train_start(self, trainer: Trainer) -> None:
        self.device = trainer.args.device

        # Handles a bug with loading optimizer states from checkpoint. Solution found
        # here: https://github.com/pytorch/pytorch/issues/2830. Not sure if this would 
        # also affect scheduler states as I haven't checked.
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def to_device(self, batch: List[Any]) -> None:
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(self.device)

    def on_train_step_start(self, trainer: Trainer, batch: List[Any]) -> None:
        self.to_device(batch)

    def on_val_step_start(self, trainer: Trainer, batch: List[Any]) -> None:
        self.to_device(batch)

    def state_dict(self) -> Dict[str, Any]:
        return {"device": self.device}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.device = state["device"]


class TQDMLogger(Callback):
    """
    """
    def __init__(self):
        # The key/value pairs to include in the postfix of the TQDM logger.
        self.postfix: Dict[str, float] = {} 
        self.buffer: int = 1 # The buffer set by `log_every_n_steps`.

        # Other state variables
        self.curr_step = 0
        self.max_steps = -1
        self.log_every_n_steps = 1

    def log(self, trainer: Trainer, k: str, v: float, to_pbar: bool = True):
        if to_pbar: self.postfix[k] = v

    def _step(self) -> None:
        if (self.curr_step % self.log_every_n_steps == 0) or (self.curr_step == self.max_steps):
            self.pbar.update(self.buffer)
            self.pbar.set_postfix(self.postfix)
            self.postfix = {}
            self.buffer = 1
        else:
            self.buffer += 1

    def on_train_start(self, trainer: Trainer) -> None:
        self.log_every_n_steps = trainer.args.log_every_n_steps

    def on_train_epoch_start(self, trainer: Trainer) -> None:
        desc = f"Training | Epoch {trainer.state.curr_epoch}/{trainer.args.max_epochs} "
        self.max_steps = trainer.max_train_steps
        self.pbar = tqdm(total = self.max_steps, desc = desc, leave = True)

    def on_train_step_end(self, trainer: Trainer, output: Any) -> None:
        self.curr_step = trainer.state.curr_step
        self._step()

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        self.pbar.close()
        self.buffer = 1

    def on_val_epoch_start(self, trainer: Trainer) -> None:
        desc = f"Validation | Epoch {trainer.state.curr_epoch}/{trainer.args.max_epochs} "
        self.max_steps = trainer.max_val_steps
        self.pbar = tqdm(total = self.max_steps, desc = desc, leave = True)

    def on_val_step_end(self, trainer: Trainer, output: Any) -> None:
        self.curr_step = trainer.state.curr_step
        self._step()

    def on_val_epoch_end(self, trainer: Trainer) -> None:
        self.pbar.close()
        self.bufer = 1


class MetricTracker(Callback):
    """
    """
    def __init__(self):
        self.metrics_step: Dict[str, List[Any]] = defaultdict(list)

    def log(self, trainer: Trainer, k: str, v: float, to_pbar: bool = True):
        # Track the training mode, current epoch, and current step along with
        # every metric iter for full reproducability. This is redundent, but 
        # for my purpose rightfuly so.
        self.metrics_step[k].append(
            [trainer.state.mode, trainer.state.curr_epoch, trainer.state.curr_step, v]
        )

    def on_train_start(self, trainer: Trainer) -> None:
        self.output_dir = os.path.join(trainer.args.output_dir, "metrics/")

        # Because of the *strict* way that the Trainer handles `output_dir`,
        # these should never get raised. 
        assert not os.path.exists(self.output_dir)
        assert os.path.exists(trainer.args.output_dir)

        # The trainer already handles verifying parent directory.
        os.makedirs(self.output_dir) 

    def flush(self) -> None:
        for k, v in self.metrics_step.items():
            path = os.path.join(self.output_dir, slugify(k) + ".csv")

            # If the file already exists, we merely append on the new values
            # rather than also including the columns again.
            columns: Optional[List[str]] = None
            if not os.path.exists(path):
                columns = ["mode", "epoch", "step", k]

            write_to_csv(path, v, columns)

    def on_epoch_start(self, trainer: Trainer) -> None:
        # Clear the metrics on every epoch start. This is viable because the
        # metrics are flushed on epoch end, and thus no data is lost.
        self.metrics_step.clear()

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.flush() 

    def on_keyboard_interrupt(self, trainer: Trainer) -> None:
        self.flush() # Make sure to flush incase of an interupt.

    def on_exception(self, trainer: Trainer, e: Exception) -> None:
        self.flush() # Make sure to flush incase of an interupt.


class SummaryWriter(Callback):
    """
    """
    def __init__(self):
        self.summary: Dict[str, List[Any]] = defaultdict(list)

    def log(self, trainer: Trainer, k: str, v: float, to_pbar: bool = True):
        self.summary[k].append(v)

    def _log_dict(self, msg: Dict, prefix: str = "") -> None:
        logger.info(prefix + " | ".join([f"{k}: {v}" for k, v in msg.items()]))

    def on_train_start(self, trainer: Trainer) -> None:
        self.start_train_time = time.perf_counter()

    def on_train_epoch_start(self, trainer: Trainer) -> None:
        self.summary.clear()
        self.start_epoch_time = time.perf_counter()

    def _build_msg(self, trainer: Trainer) -> Dict:
        epoch = trainer.state.curr_epoch
        max_epochs = trainer.args.max_epochs

        msg = {
            "Epoch": f"[{epoch:>{len(str(max_epochs))}}/{max_epochs}]",
            "Total Time": f"{time.perf_counter()-self.start_train_time:8.4f}",
            "Epoch Time": f"{time.perf_counter()-self.start_epoch_time:8.4f}",
        }
        for k, v in self.summary.items():
            mv = 0
            if len(v) != 0:
                mv = sum(v) / len(v)

            msg[f"Mean {k}"] = f"{mv:.10f}"

        return msg

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        self._log_dict(self._build_msg(trainer), "training summary - ")

    def on_val_epoch_start(self, trainer: Trainer) -> None:
        self.summary.clear()
        self.start_epoch_time = time.perf_counter()

    def on_val_epoch_end(self, trainer: Trainer) -> None:
        self._log_dict(self._build_msg(trainer), "validation summary - ")



class ModelCheckpoint(Callback):
    """
    """
    def __init__(self, monitor: str, mode: str = "min") -> None:
        self.monitor = monitor
        self.mode = mode

        self.filename = f"model_best_{slugify(self.monitor)}.pt"
        self.best_metric: Optional[float] = None
        self.curr_epoch: List[float] = [] # Holds the log for the current epoch.

    def log(self, trainer: Trainer, k: str, v: float, to_pbar: bool = True) -> None:
        """This is the logic that tracks the current values for `monitor`. Do
        note that this implementation does NOT: 
         - Handle logging the same metric twice.
         - Handle *well* the case in which no metric matches `monitor`.
        """
        if k == self.monitor:
            self.curr_epoch.append(v)

    def on_epoch_start(self, trainer: Trainer) -> None:
        self.curr_epoch.clear()

    def on_epoch_end(self, trainer: Trainer) -> None:
        # Handle the case in which an empty 
        if len(self.curr_epoch) == 0:
            logger.warning(
                f"It seems no metrics matching '{self.monitor}' were found and "
                f"are being tracked. Training will continue, but ModelCheckpoint "
                f"will not be saving any checkpoints for this metric."
            )
            return

        value = sum(self.curr_epoch) / len(self.curr_epoch)
        save_checkpoint = (
            (self.best_metric is None) or 
            (self.mode == "min" and value < self.best_metric) or 
            (self.mode == "max" and value > self.best_metric)
        )
        if save_checkpoint:
            self.best_metric = value
            self._save_model(trainer)

    def _save_model(self, trainer: Trainer) -> None:
        logger.info(f"Saving model checkpoint (based on *{self.monitor}*)")
        trainer.save_checkpoint(self.filename)


class SaveFinalModel(Callback):
    def on_train_end(self, trainer: Trainer) -> None:
        logger.info(f"Saving final model checkpoint")
        trainer.save_checkpoint("model_final.pt")

    def on_keyboard_interrupt(self, trainer: Trainer) -> None:
        self.on_train_end(trainer)

    def on_exception(self, trainer: Trainer, e: Exception) -> None:
        self.on_train_end(trainer)


################################################################################
# Utils
################################################################################

def slugify(value: str, allow_unicode: bool = False) -> str:
    """Slightly modified version of slugify taken from the Django repo (see
    links below). Modified to sub illegal values with '-' instead. 

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    
    Resources:
     - https://github.com/django/django/blob/master/django/utils/text.py
     - https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename

    """
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")

    value = re.sub(r"[^\w\s-]", "-", value.lower())
    value = re.sub(r"[-\s]+", "-", value).strip("-_")

    return value


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_to_csv(filename: str, rows: List[List[Any]], columns: Optional[List[str]] = None, mode: str = "a") -> None:
    """ """
    with open(filename, mode, newline = "") as f:
        writer = csv.writer(f)
        if columns is not None:
            writer.writerow(columns)
        writer.writerows(rows)


