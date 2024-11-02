"""Flowers102 dataset loader implementation."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .base_dataloader import BaseDataset

logger = logging.getLogger(__name__)


class Flowers102DataModule:
    """Data module for Flowers102 dataset."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        train_split_pct: float = 1.0,
        seeds: Optional[List[int]] = None,
        batch_sizes: Tuple[int, int, int] = (96, 64, 64),
        num_workers: int = 4,
        download: bool = True,
        force_preprocess: bool = False,
        resize_size: int = 232,
        crop_size: int = 224,
    ):
        """Initialize data module."""
        self.data_dir = Path(data_dir)
        self.train_split_pct = train_split_pct
        self.seeds = seeds or [42]
        self.train_batch_size, self.val_batch_size, self.test_batch_size = batch_sizes
        self.num_workers = num_workers
        self.download = download
        self.force_preprocess = force_preprocess
        self.resize_size = resize_size
        self.crop_size = crop_size

        # Create transforms
        self.transform = self._create_transforms()

        # Download and prepare dataset if needed
        if download:
            self._download_dataset()

    def _create_transforms(self) -> T.Compose:
        """Create data transforms.

        Returns:
            Composition of transforms
        """
        return T.Compose(
            [
                T.Resize(self.resize_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _download_dataset(self) -> None:
        """Download the dataset."""
        try:
            # Download using torchvision's built-in functionality
            for split in ["train", "val", "test"]:
                torchvision.datasets.Flowers102(
                    root=self.data_dir, split=split, download=True
                )
            logger.info("Dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise

    def create_data_loaders(
        self,
    ) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        """Create data loaders for all splits.

        Returns:
            Tuple of (train_loaders, val_loaders, test_loader)
        """
        train_loaders = []
        val_loaders = []

        # Create multiple train/val splits based on seeds
        for seed in self.seeds:
            train_dataset, val_dataset = self._create_train_val_datasets(seed)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        # Create test loader
        test_dataset = self._create_test_dataset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loaders, val_loaders, test_loader

    def _create_train_val_datasets(self, seed: int) -> Tuple[BaseDataset, BaseDataset]:
        """Create training and validation datasets.

        Args:
            seed: Random seed for split

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load the original training data
        dataset = torchvision.datasets.Flowers102(
            root=self.data_dir, split="train", download=False
        )

        # Get image paths and labels
        image_paths = np.array(dataset._image_files)
        labels = torch.tensor(dataset._labels)

        # Create train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(image_paths))
        split = int(self.train_split_pct * len(indices))

        train_indices = indices[:split]
        val_indices = indices[split:]

        # Create datasets
        train_dataset = BaseDataset(
            image_paths[train_indices],
            labels[train_indices],
            self.transform,
            resize_size=self.resize_size,
            force_preprocess=self.force_preprocess,
        )

        val_dataset = BaseDataset(
            image_paths[val_indices],
            labels[val_indices],
            self.transform,
            resize_size=self.resize_size,
            force_preprocess=self.force_preprocess,
        )

        return train_dataset, val_dataset

    def _create_test_dataset(self) -> BaseDataset:
        """Create test dataset.

        Returns:
            Test dataset
        """
        dataset = torchvision.datasets.Flowers102(
            root=self.data_dir, split="test", download=False
        )

        return BaseDataset(
            np.array(dataset._image_files),
            torch.tensor(dataset._labels),
            self.transform,
            resize_size=self.resize_size,
            force_preprocess=self.force_preprocess,
        )


def create_dataloaders(
    train_split_pct: float = 1.0,
    seeds: Optional[List[int]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
    force_preprocess: bool = True,
    resize_size: int = 232,
    crop_size: int = 224,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create data loaders for Flowers102.

    Args:
        train_split_pct: Percentage of training data to use
        seeds: List of random seeds
        data_dir: Directory containing the dataset
        download: Whether to download the dataset if not found
        force_preprocess: Whether to force image preprocessing
        resize_size: Size to resize images to
        crop_size: Size to crop images to

    Returns:
        Tuple of (train_loaders, val_loaders, test_loader)
    """
    if data_dir is None:
        data_dir = Path.cwd() / "flowers"

    data_module = Flowers102DataModule(
        data_dir=data_dir,
        train_split_pct=train_split_pct,
        seeds=seeds,
        download=download,
        force_preprocess=force_preprocess,
        resize_size=resize_size,
        crop_size=crop_size,
    )

    return data_module.create_data_loaders()


# Export the function
__all__ = ["create_dataloaders", "Flowers102DataModule"]
