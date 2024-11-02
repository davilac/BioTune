"""Custom metrics and metric utilities for training."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchmetrics


class MetricsCalculator:
    """Calculator for multiple metrics during training.

    Args:
        num_classes: Number of classes for classification metrics
        device: Device to place metrics on
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.metrics = {
            'accuracy': torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_classes,
                average="macro"
            ).to(device),
            'precision': torchmetrics.Precision(
                task="multiclass",
                num_classes=num_classes,
                average="macro"
            ).to(device),
            'recall': torchmetrics.Recall(
                task="multiclass",
                num_classes=num_classes,
                average="macro"
            ).to(device),
            'f1': torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_classes,
                average="macro"
            ).to(device),
            'kappa': torchmetrics.CohenKappa(
                task="multiclass",
                num_classes=num_classes
            ).to(device)
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics with new predictions.

        Args:
            preds: Model predictions
            targets: Ground truth labels
        """
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        return {name: metric.compute().item() 
                for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


class WeightedLoss:
    """Weighted loss function wrapper.

    Args:
        base_criterion: Base loss function
        weight: Optional tensor of weights
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        weight: Optional[torch.Tensor] = None
    ):
        self.base_criterion = base_criterion
        self.weight = weight

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted loss.

        Args:
            pred: Model predictions
            target: Ground truth labels

        Returns:
            Weighted loss value
        """
        base_loss = self.base_criterion(pred, target)
        if self.weight is not None:
            base_loss = base_loss * self.weight[target]
        return base_loss.mean()
