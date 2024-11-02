"""Utilities for model initialization and management."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from src.models.lora import (LoRADenseNet121, LoRAInceptionV3, LoRAResNet50,
                           LoRAVGG19)


class RGNScheduler(_LRScheduler):
    """Learning rate scheduler based on Relative Gradient Norm.

    Args:
        optimizer: The optimizer to adjust learning rates for
        ratios: Initial learning rate ratios
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        ratios: List[float],
        last_epoch: int = -1
    ):
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.ratios = ratios
        super().__init__(optimizer, last_epoch, verbose=False)

    def get_lr(self) -> List[float]:
        """Calculate new learning rates.

        Returns:
            List of new learning rates
        """
        if len(self.base_lrs) != len(self.ratios):
            raise ValueError(
                f"LR dimension: {len(self.base_lrs)} doesn't match "
                f"ratios dimension: {len(self.ratios)}"
            )
        return [base_lr * r for base_lr, r in zip(self.base_lrs, self.ratios)]


def set_selective_grads(
    model: nn.Module,
    trainable_blocks_id: List[int],
    network_type: str = "resnet50",
    load_type: str = "block-wise"
) -> None:
    """Set trainable parameters for selective fine-tuning.

    Args:
        model: The model to modify
        trainable_blocks_id: List of block IDs to train
        network_type: Type of network architecture
        load_type: Type of loading strategy ('block-wise' or 'layer-wise')
    """
    set_id_map = create_set_ids(network_type=network_type, method=load_type)

    trainable_layers_id = []
    for block_id in trainable_blocks_id:
        trainable_layers_id.extend(
            range(
                set_id_map[block_id][0],
                set_id_map[block_id][1] + 1,
            )
        )

    for param_id, named_param in enumerate(model.named_parameters()):
        param = named_param[1]
        param.requires_grad = param_id in trainable_layers_id


def create_set_ids(network_type: str = "resnet50", method: str = "layer-wise") -> List[List[int]]:
    """Create mapping of layer/block IDs for different architectures.

    Args:
        network_type: Type of network architecture
        method: Type of loading strategy

    Returns:
        List of ID ranges for each set

    Raises:
        ValueError: If network type is not supported
    """
    network_configs = {
        "resnet50": {
            "layer-wise": lambda: [[a, b] for a, b in zip(range(0, 159, 3), range(2, 159, 3))] + [[159, 160]],
            "block-wise": lambda: [[0, 2], [3, 32], [33, 71], [72, 128], [129, 158], [159, 160]]
        },
        "densenet121": {
            "layer-wise": _create_densenet_layer_ids,
            "block-wise": lambda: [[0, 2], [3, 38], [39, 41], [42, 113], [114, 116],
                                 [117, 260], [261, 263], [264, 359], [360, 363]]
        },
        "vgg19": {
            "layer-wise": lambda: [[a, b] for a, b in zip(range(0, 38, 2), range(1, 38, 2))],
            "block-wise": lambda: [[0, 3], [4, 7], [8, 15], [16, 23], [24, 31], [32, 37]]
        },
        "inception_v3": {
            "layer-wise": _create_inception_layer_ids,
            "block-wise": lambda: [[0, 14], [15, 35], [36, 56], [57, 77], [78, 89],
                                 [90, 119], [120, 149], [150, 179], [180, 209],
                                 [210, 217], [218, 235], [236, 262], [263, 289],
                                 [290, 291]]
        }
    }

    if network_type not in network_configs:
        raise ValueError(f"Network {network_type} not found")

    if method not in network_configs[network_type]:
        raise ValueError(f"Method {method} not found for network {network_type}")

    set_id_map = network_configs[network_type][method]()
    print(f"[{network_type}] Number of {method} sets: {len(set_id_map)}")
    
    return set_id_map


def _create_densenet_layer_ids() -> List[List[int]]:
    """Create layer-wise IDs for DenseNet."""
    layers = []
    current = 0
    
    # Initial layers
    layers.extend([[0, 2], [3, 8]])
    current = 9
    
    # Dense blocks and transitions
    for _ in range(4):
        for _ in range(6):
            layers.append([current, current + 5])
            current += 6
        layers.append([current, current + 2])
        current += 3
    
    # Final layers
    layers.append([current, current + 3])
    
    return layers


def _create_inception_layer_ids() -> List[List[int]]:
    """Create layer-wise IDs for Inception."""
    base_layers = [[a, b] for a, b in zip(range(0, 216, 3), range(2, 216, 3))]
    base_layers.append([216, 217])
    
    tail_layers = [[a, b] for a, b in zip(range(218, 290, 3), range(220, 290, 3))]
    tail_layers.append([290, 291])
    
    return base_layers + tail_layers


def load_model(
    network: str,
    ft_strategy: str,
    output_classes: int,
    trainable_set_ids: Optional[List[int]] = None,
) -> nn.Module:
    """Load and configure a model for fine-tuning.

    Args:
        network: Type of network architecture
        ft_strategy: Fine-tuning strategy
        output_classes: Number of output classes
        trainable_set_ids: List of trainable set IDs

    Returns:
        Configured model

    Raises:
        ValueError: If network type or strategy is not supported
    """
    model_loaders = {
        "resnet50": load_resnet50,
        "densenet121": load_densenet121,
        "vgg19": load_vgg19,
        "inception_v3": load_inceptionv3
    }

    if network not in model_loaders:
        raise ValueError(f"Network {network} not found")

    return model_loaders[network](ft_strategy, output_classes, trainable_set_ids)


def generate_model(
    seed: int,
    network: str = "resnet50",
    ft_method: str = "ft_full",
    learning_rate: float = 0.05,
    trainable_set_ids: Optional[List[int]] = None,
    lr_ratios: Optional[Dict[str, float]] = None,
    output_classes: int = 10,
    device: str = "cuda:0",
) -> Tuple[nn.Module, optim.Optimizer, Optional[_LRScheduler]]:
    """Generate model, optimizer, and scheduler for training.

    Args:
        seed: Random seed
        network: Type of network architecture
        ft_method: Fine-tuning method
        learning_rate: Initial learning rate
        trainable_set_ids: List of trainable set IDs
        lr_ratios: Learning rate ratios for different layers
        output_classes: Number of output classes
        device: Device to place model on

    Returns:
        Tuple of (model, optimizer, scheduler)
    """
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load model
    model = load_model(
        network=network,
        ft_strategy=ft_method,
        output_classes=output_classes,
        trainable_set_ids=trainable_set_ids,
    )
    model = model.to(device)

    # Configure optimizer
    if ft_method in ["autoRGN", "adaptive_block_rgn"]:
        param_groups = _create_param_groups(model, network, learning_rate)
        optimizer = optim.Adam(param_groups, lr=learning_rate)
        scheduler = RGNScheduler(optimizer, len(optimizer.param_groups) * [1])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    return model, optimizer, scheduler


def _create_param_groups(
    model: nn.Module,
    network: str,
    learning_rate: float
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Args:
        model: The model
        network: Type of network architecture
        learning_rate: Initial learning rate

    Returns:
        List of parameter group dictionaries
    """
    param_group_configs = {
        "resnet50": lambda m: [
            {"params": module.parameters(), "lr": learning_rate}
            for name, module in m.named_modules()
            if ("conv" in name or "bn" in name or "fc" in name or
                "downsample.0" in name or "downsample.1" in name)
        ],
        "densenet121": lambda m: [
            {"params": module.parameters(), "lr": learning_rate}
            for name, module in m.named_modules()
            if ("conv" in name or "norm" in name or "classifier" in name)
        ],
        "vgg19": lambda m: [
            {"params": module.parameters(), "lr": learning_rate}
            for name, module in m.named_modules()
            if name in [
                "features.0", "features.2", "features.5", "features.7",
                "features.10", "features.12", "features.14", "features.16",
                "features.19", "features.21", "features.23", "features.25",
                "features.28", "features.30", "features.32", "features.34",
                "classifier.0", "classifier.3", "classifier.6",
            ]
        ],
        "inception_v3": lambda m: [
            {"params": module.parameters(), "lr": learning_rate}
            for name, module in m.named_modules()
            if name.endswith(".conv") or name.endswith(".bn") or
               name.endswith(".fc") or name == "fc"
        ]
    }

    if network not in param_group_configs:
        raise ValueError(f"Network {network} not supported for parameter grouping")

    return param_group_configs[network](model)
