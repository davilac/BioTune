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
        """Initialize data module.
        
        Args:
            train_split_pct: Percentage of data per fold (not random %, but stratified fold size)
                - 1.0 = 1 fold with 100% of data
                - 0.5 = 2 folds with 50% each (fold 0: 0-50%, fold 1: 50-100%)
                - 0.25 = 4 folds with 25% each
            seeds: Random seeds for training runs (not used for fold creation)
        """
        self.data_dir = Path(data_dir)
        self.train_split_pct = train_split_pct
        self.seeds = seeds or [42]
        self.train_batch_size, self.val_batch_size, self.test_batch_size = batch_sizes
        self.num_workers = num_workers
        self.download = download
        self.force_preprocess = force_preprocess
        self.resize_size = resize_size
        self.crop_size = crop_size
        
        # Calculate number of folds
        self.n_folds = int(np.ceil(1.0 / train_split_pct))

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
        """Create data loaders for all folds.
        
        Creates n_folds stratified folds where all folds together = 100% of data.
        Example: train_split_pct=0.5 → 2 folds (0-50%, 50-100%)

        Returns:
            Tuple of (train_loaders, val_loaders, test_loader)
        """
        train_loaders = []
        val_loaders = []

        # Create stratified folds (NOT random splits)
        for fold_idx in range(self.n_folds):
            train_dataset, val_dataset = self._create_train_val_datasets_fold(fold_idx)

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
            
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}: Train={len(train_dataset)}, Val={len(val_dataset)}")

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

    def _create_train_val_datasets_fold(self, fold_idx: int) -> Tuple[BaseDataset, BaseDataset]:
        """Create training and validation datasets for a specific fold.
        
        Uses stratified sampling: each fold gets consecutive samples per class.
        Example with train_split_pct=0.5 (2 folds):
          - Fold 0: samples 0-50% from each class
          - Fold 1: samples 50-100% from each class
        
        Args:
            fold_idx: Fold index (0 to n_folds-1)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load the original training and validation data
        train_dataset = torchvision.datasets.Flowers102(
            root=self.data_dir, split="train", download=False
        )
        val_dataset = torchvision.datasets.Flowers102(
            root=self.data_dir, split="val", download=False
        )

        # Get image paths and labels
        train_images = np.array(train_dataset._image_files)
        train_labels = np.array(train_dataset._labels)
        val_images = np.array(val_dataset._image_files)
        val_labels = np.array(val_dataset._labels)
        
        # Get number of classes
        n_classes = 102  # Flowers102 has 102 classes
        
        # Extract indices per class (stratified)
        train_class_indices = [
            np.where(train_labels == i)[0] for i in range(n_classes)
        ]
        val_class_indices = [
            np.where(val_labels == i)[0] for i in range(n_classes)
        ]
        
        # Calculate samples per class for this fold
        size_train_per_class = int(self.train_split_pct * len(train_class_indices[0]))
        size_val_per_class = int(self.train_split_pct * len(val_class_indices[0]))
        
        # Get indices for this fold (stratified sampling)
        if fold_idx == self.n_folds - 1:
            # Last fold gets remaining samples
            train_fold_indices = np.array([
                indices[fold_idx * size_train_per_class:]
                for indices in train_class_indices
            ]).flatten()
            val_fold_indices = np.array([
                indices[fold_idx * size_val_per_class:]
                for indices in val_class_indices
            ]).flatten()
        else:
            # Other folds get fixed-size chunks
            train_fold_indices = np.array([
                indices[fold_idx * size_train_per_class : (fold_idx + 1) * size_train_per_class]
                for indices in train_class_indices
            ]).flatten()
            val_fold_indices = np.array([
                indices[fold_idx * size_val_per_class : (fold_idx + 1) * size_val_per_class]
                for indices in val_class_indices
            ]).flatten()

        # Create datasets for this fold
        train_fold_dataset = BaseDataset(
            train_images[train_fold_indices],
            torch.tensor(train_labels[train_fold_indices]),
            self.transform,
            resize_size=self.resize_size,
            force_preprocess=self.force_preprocess,
        )

        val_fold_dataset = BaseDataset(
            val_images[val_fold_indices],
            torch.tensor(val_labels[val_fold_indices]),
            self.transform,
            resize_size=self.resize_size,
            force_preprocess=self.force_preprocess,
        )

        return train_fold_dataset, val_fold_dataset

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
    """Create data loaders for Flowers102 with stratified folds.
    
    Creates n_folds = ceil(1/train_split_pct) stratified folds where:
    - All folds together cover 100% of data (no overlap)
    - Each fold is stratified (same class distribution)
    
    Examples:
        train_split_pct=1.0 → 1 fold with 100% of data
        train_split_pct=0.5 → 2 folds (fold 0: 0-50%, fold 1: 50-100%)
        train_split_pct=0.25 → 4 folds (each 25%)
    
    Generations cycle through folds:
        - 5 generations + 2 folds → [fold0, fold1, fold0, fold1, fold0]

    Args:
        train_split_pct: Percentage of data per fold (default: 1.0)
        seeds: Random seeds for training runs (NOT used for fold creation)
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
