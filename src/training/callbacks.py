"""Callback implementations for training."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.training.utils import save_model

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback implementation.
    
    Args:
        mode: 'min' or 'max' for minimizing or maximizing metric
        min_delta: Minimum change in monitored value to qualify as improvement
        patience: Number of epochs with no improvement after which training will stop
        percentage: Whether to consider min_delta as a percentage
        verbose: Whether to print early stopping messages
    """

    def __init__(
        self,
        mode: str = "min",
        min_delta: float = 0,
        patience: int = 3,
        percentage: bool = False,
        verbose: bool = False,
    ):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.percentage = percentage
        self.verbose = verbose
        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        self.is_better = None
        
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(
        self, metrics: float, model: nn.Module, ckpt_path: Union[str, Path]
    ) -> bool:
        """Update early stopping state.

        Args:
            metrics: Current epoch's metrics
            model: Model to save if metrics improved
            ckpt_path: Path to save model checkpoint

        Returns:
            True if training should stop, False otherwise
        """
        if self.best is None:
            self.best = metrics
            save_model(model=model, ckpt_path=ckpt_path)
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            save_model(model=model, ckpt_path=ckpt_path)
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            if self.verbose:
                logger.info("Stopping early due to no improvement in metrics")
            return True

        return False

    def _init_is_better(self, mode: str, min_delta: float, percentage: bool) -> None:
        """Initialize the is_better function based on mode and parameters.

        Args:
            mode: 'min' or 'max'
            min_delta: Minimum change in monitored value
            percentage: Whether to consider min_delta as a percentage
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")

        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            else:
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            else:
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class ModelCheckpoint:
    """Save model checkpoints based on various criteria.

    Args:
        dirpath: Directory to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max' for minimizing or maximizing metric
        save_top_k: Number of best models to save
        filename: Format string for checkpoint filenames
    """

    def __init__(
        self,
        dirpath: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        filename: str = "model-{epoch:02d}-{val_loss:.2f}",
    ):
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        
        self.best_k_models = {}
        self.current_best = None
        
        # Create checkpoint directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self, model: nn.Module, metrics: dict, epoch: int
    ) -> None:
        """Save model checkpoint if metrics improved.

        Args:
            model: Model to save
            metrics: Current metrics
            epoch: Current epoch
        """
        monitor_value = metrics[self.monitor]
        
        filename = self.filename.format(epoch=epoch, **metrics)
        filepath = self.dirpath / filename

        if self.current_best is None:
            self.current_best = monitor_value
            self.save_model(model, filepath)
            self.best_k_models[filepath] = monitor_value
        else:
            if (self.mode == "min" and monitor_value < self.current_best) or \
               (self.mode == "max" and monitor_value > self.current_best):
                self.current_best = monitor_value
                self.save_model(model, filepath)
                self.best_k_models[filepath] = monitor_value
                
                # Remove older models if we have too many
                if len(self.best_k_models) > self.save_top_k:
                    worst_filepath = min(
                        self.best_k_models.items(),
                        key=lambda x: x[1] if self.mode == "max" else -x[1]
                    )[0]
                    Path(worst_filepath).unlink(missing_ok=True)
                    del self.best_k_models[worst_filepath]

    def save_model(self, model: nn.Module, filepath: Path) -> None:
        """Save model to file.

        Args:
            model: Model to save
            filepath: Path to save model to
        """
        save_model(model=model, ckpt_path=filepath)
