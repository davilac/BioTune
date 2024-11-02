"""General utility functions for training and model management."""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_model(model: nn.Module, ckpt_path: Union[str, Path]) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        ckpt_path: Path to save checkpoint
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        # logger.info(f"Model saved to {ckpt_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")


def load_saved_model(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Load saved model checkpoint.

    Args:
        model: Model to load weights into
        ckpt_path: Path to checkpoint
        device: Device to load model to

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Model loaded from {ckpt_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def seed_worker(worker_id: int) -> None:
    """Set random seed for data loader workers.

    Args:
        worker_id: ID of the worker
    """
    import numpy as np

    np.random.seed(np.random.get_state()[1][0] + worker_id)


def print_model_info(model: nn.Module, detailed: bool = False) -> None:
    """Print model information.

    Args:
        model: Model to print information for
        detailed: Whether to print detailed parameter information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")

    if detailed:
        print("\nTrainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} parameters")


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device.

    Args:
        device_str: Optional device string (e.g., 'cuda:0', 'cpu')

    Returns:
        torch.device
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    return device


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update meter with new value.

        Args:
            val: Value to add
            n: Number of items the value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
