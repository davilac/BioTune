"""Main training logic for model training and evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torchmetrics

from src.training.callbacks import EarlyStopping
from src.training.utils import save_model, load_saved_model

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class for model training and evaluation.

    Args:
        model: The neural network model to train
        optimizer: The optimizer for training
        scheduler: Optional learning rate scheduler
        loss_fn: Loss function
        device: Device to use for training
        num_classes: Number of classes for metrics
        save_dir: Directory to save model checkpoints
        network_type: Type of network architecture
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: nn.Module,
        device: torch.device,
        num_classes: int,
        save_dir: Union[str, Path],
        network_type: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = Path(save_dir)
        self.network_type = network_type

        # Initialize metrics
        self.metrics = self._initialize_metrics(num_classes)

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_metrics(
        self, num_classes: int
    ) -> Dict[str, Dict[str, torchmetrics.Metric]]:
        """Initialize training metrics."""
        metric_types = {
            "train": {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro"
                ).to(self.device)
            },
            "val": {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro"
                ).to(self.device)
            },
            "test": {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro"
                ).to(self.device),
                "avg_precision": torchmetrics.AveragePrecision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ).to(self.device),
                "kappa": torchmetrics.CohenKappa(
                    task="multiclass", num_classes=num_classes
                ).to(self.device),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ).to(self.device),
            },
        }
        return metric_types

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            if self.network_type == "inception_v3":
                loss = self.loss_fn(outputs[0], labels)
                outputs = outputs[0]
            else:
                loss = self.loss_fn(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.metrics["train"]["accuracy"](outputs, labels)

        avg_loss = total_loss / len(train_loader)
        accuracy = self.metrics["train"]["accuracy"].compute()
        self.metrics["train"]["accuracy"].reset()

        return avg_loss, accuracy.item()

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                if self.network_type == "inception_v3":
                    loss = self.loss_fn(outputs[0], labels)
                    outputs = outputs[0]
                else:
                    loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()
                self.metrics["val"]["accuracy"](outputs, labels)

        avg_loss = total_loss / len(val_loader)
        accuracy = self.metrics["val"]["accuracy"].compute()
        self.metrics["val"]["accuracy"].reset()

        return avg_loss, accuracy.item()

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                if self.network_type == "inception_v3":
                    loss = self.loss_fn(outputs[0], labels)
                    outputs = outputs[0]
                else:
                    loss = self.loss_fn(outputs, labels)

                outputs = torch.softmax(outputs, dim=1)
                total_loss += loss.item()

                # Update all test metrics
                for metric in self.metrics["test"].values():
                    metric(outputs, labels)

        # Compute final metrics
        results = {
            "loss": total_loss / len(test_loader),
            "accuracy": self.metrics["test"]["accuracy"].compute().item(),
            "avg_precision": self.metrics["test"]["avg_precision"].compute().item(),
            "kappa": self.metrics["test"]["kappa"].compute().item(),
            "f1": self.metrics["test"]["f1"].compute().item(),
        }

        # Reset metrics
        for metric in self.metrics["test"].values():
            metric.reset()

        return results

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        early_stop_patience: int = 5,
        save_weight_grads: bool = False,
    ) -> Tuple[float, float, int, nn.Module, List[Dict], List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            num_epochs: Number of epochs to train
            early_stop_patience: Patience for early stopping
            save_weight_grads: Whether to save weight gradients

        Returns:
            Tuple of (test_accuracy, best_val_accuracy, best_epoch, best_model,
                    rgn_ratios_per_epoch, metrics)
        """
        early_stopping = EarlyStopping(
            patience=early_stop_patience,
            mode="max",
            verbose=True,
        )

        best_val_acc = 0.0
        best_epoch = 0
        best_metrics = []  # Initialize with default values
        rgn_ratios_per_epoch = []

        # Initialize best_metrics with default values
        best_metrics = [0.0] * 9  # Initialize with 9 zeros for all metrics

        logger.info("\nStarting Training:")
        logger.info(
            f"{'Epoch':^7} {'Train Loss':^12} {'Train Acc':^12} {'Val Loss':^12} {'Val Acc':^12}"
        )
        logger.info("-" * 60)

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch results
            logger.info(
                f"{epoch + 1:^7d} "
                f"{train_loss:^12.4f} "
                f"{train_acc:^12.4f} "
                f"{val_loss:^12.4f} "
                f"{val_acc:^12.4f}"
            )

            # Save best model and update best metrics
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_model(self.model, self.save_dir / "best_model.pt")

                if test_loader is not None:
                    test_metrics = self.test(test_loader)
                    best_metrics = [
                        train_loss,  # 0: train loss
                        val_loss,  # 1: val loss
                        train_acc,  # 2: train accuracy
                        val_acc,  # 3: val accuracy
                        test_metrics["loss"],  # 4: test loss
                        test_metrics["accuracy"],  # 5: test accuracy
                        test_metrics["avg_precision"],  # 6: test average precision
                        test_metrics["kappa"],  # 7: test kappa
                        test_metrics["f1"],  # 8: test f1
                    ]
                else:
                    # If no test loader, still maintain the list structure with training and validation metrics
                    best_metrics = [
                        train_loss,  # 0: train loss
                        val_loss,  # 1: val loss
                        train_acc,  # 2: train accuracy
                        val_acc,  # 3: val accuracy
                        0.0,  # 4: placeholder for test loss
                        0.0,  # 5: placeholder for test accuracy
                        0.0,  # 6: placeholder for test average precision
                        0.0,  # 7: placeholder for test kappa
                        0.0,  # 8: placeholder for test f1
                    ]

            # Early stopping check
            if early_stopping.step(
                val_acc, self.model, self.save_dir / "checkpoint.pt"
            ):
                logger.info(f"\nEarly stopping triggered after epoch {epoch + 1}")
                break

            # Save RGN ratios if requested
            if save_weight_grads:
                rgn_ratios = self._compute_rgn_ratios()
                rgn_ratios_per_epoch.append(rgn_ratios)

        logger.info("\nTraining completed!")
        logger.info(
            f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}"
        )

        # Load best model for final testing
        best_model = load_saved_model(
            self.model, self.save_dir / "best_model.pt", self.device
        )

        if test_loader is not None:
            test_metrics = self.test(test_loader)
            test_acc = test_metrics["accuracy"]
            logger.info(
                f"\nFinal test metrics:\n"
                f"Accuracy: {test_metrics['accuracy']:.4f}\n"
                f"F1 Score: {test_metrics['f1']:.4f}\n"
                f"Kappa: {test_metrics['kappa']:.4f}\n"
                f"Average Precision: {test_metrics['avg_precision']:.4f}"
            )
        else:
            test_acc = 0.0

        return (
            test_acc,
            best_val_acc,
            best_epoch,
            best_model,
            rgn_ratios_per_epoch,
            best_metrics,  # This now always has 9 elements
        )

    def _compute_rgn_ratios(self) -> Dict[str, float]:
        """Compute relative gradient norms for model parameters."""
        rgn_ratios = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                weight_norm = torch.norm(param).item()
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    rgn_ratios[name] = grad_norm / weight_norm
                else:
                    rgn_ratios[name] = None
        return rgn_ratios
